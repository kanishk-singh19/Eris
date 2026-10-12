#!/usr/bin/env python3
"""
MarginNote: generate the printed side-note of a historical German passage (unseen authors).

Approach
--------
* A small pretrained causal LM (Qwen2.5-0.5B, public Hugging Face weights, byte-level BPE so every
  historical character is representable) is fine-tuned with LoRA adapters on the 8,200 training
  passages: prompt = the whole paragraph with the anchor marker, target = the printed note.
* Spelling: the scorer NFKC-normalises both strings, which already maps the long s (ſ) to s and
  expands ligatures, so the model sees NFKC text. The historical "e above" mark (oͤ) is folded into
  modern umlauts for the model and restored afterwards when the passage itself uses that convention
  (a row-local decision: only that row's own context is consulted).
* Decoding: beam search plus 16 samples per row; the final note is chosen by minimum-Bayes-risk
  decoding with chrF (the competition metric) as utility, i.e. the candidate that agrees most with
  the other candidates. This directly optimises the expected character n-gram F-score.
* Training happens entirely inside this script (fixed plan: 3 epochs, no runtime-dependent
  switches). No test row is used for anything except its own prediction.

Usage: python solution.py <public_dataset_dir> <submission_csv_path>
"""
import os, sys, math, time, random, unicodedata, re
from collections import Counter
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

DATA_DIR, SUB_PATH = sys.argv[1], sys.argv[2]
MODEL = 'Qwen/Qwen2.5-0.5B'
SEED = 0
EPOCHS = 3
LR = 2e-4
RANK, ALPHA = 32, 64
BS, ACCUM = 8, 2
MAXLEN = 640
NSAMP, BEAMS, TEMP = 16, 4, 0.8
GEN_BS = 32

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', dev, flush=True)

# ----------------------------------------------------------------------------------------------
# chrF (character n-gram F-score, n=1..6, beta=2) on NFKC-lowercased, whitespace-free strings
# ----------------------------------------------------------------------------------------------
def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    return ''.join(s.split())

def chrf(hyp, ref, N=6, beta=2.0):
    h, r = norm(hyp), norm(ref)
    if not h or not r:
        return 0.0
    precs, recs = [], []
    for n in range(1, N + 1):
        hg = Counter(h[i:i+n] for i in range(len(h)-n+1)); rg = Counter(r[i:i+n] for i in range(len(r)-n+1))
        th, trr = sum(hg.values()), sum(rg.values())
        if th == 0 or trr == 0:
            continue
        m = sum((hg & rg).values()); precs.append(m/th); recs.append(m/trr)
    if not precs:
        return 0.0
    P, R = sum(precs)/len(precs), sum(recs)/len(recs)
    if P + R == 0:
        return 0.0
    b2 = beta*beta
    return (1+b2)*P*R/(b2*P+R)

# ----------------------------------------------------------------------------------------------
# Spelling canonicalisation (model side) and row-local restoration
# ----------------------------------------------------------------------------------------------
COMB = 'ͤ'   # combining Latin small letter e (historical umlaut mark)
UML = {'a'+COMB: 'ä', 'o'+COMB: 'ö', 'u'+COMB: 'ü', 'A'+COMB: 'Ä', 'O'+COMB: 'Ö', 'U'+COMB: 'Ü'}
UML_BACK = {v: k for k, v in UML.items()}
ANCHOR = '⟦NOTE⟧'

def canon(s):
    s = unicodedata.normalize('NFKC', str(s))      # same normalisation as the scorer (ſ -> s etc.)
    for k, v in UML.items():
        s = s.replace(k, v)
    return s

def restore(note, ctx):
    """If THIS passage writes umlauts with the e-above mark (and never with modern umlauts), the
    note printed beside it does too, so map the model's ä/ö/ü back. Uses only the row's own context."""
    if COMB in ctx and re.search('[äöüÄÖÜ]', ctx) is None:
        for k, v in UML_BACK.items():
            note = note.replace(k, v)
    return note

# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
tr = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
te = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
print('train', len(tr), 'test', len(te), flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = 'left'

def prompt(ctx):
    return 'Text: ' + canon(ctx) + '\nRandnote:'

def encode_train(ctx, note):
    pi = tok(prompt(ctx)).input_ids
    ti = tok(' ' + canon(note)).input_ids + [tok.eos_token_id]
    pi = pi[-(MAXLEN - len(ti)):]
    return pi + ti, [-100]*len(pi) + ti

train_enc = [encode_train(c, n) for c, n in zip(tr.context, tr.note)]
print('max train len', max(len(x[0]) for x in train_enc), flush=True)

# ----------------------------------------------------------------------------------------------
# Model + LoRA (plain PyTorch implementation, no extra dependency)
# ----------------------------------------------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
for q in model.parameters():
    q.requires_grad_(False)

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.05):
        super().__init__()
        self.base = base; self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        y = self.base(x)
        z = F.linear(F.linear(self.drop(x).float(), self.A), self.B) * self.scale
        return y + z.to(y.dtype)

TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
n_lora = 0
for name, mod in list(model.named_modules()):
    for child_name, child in list(mod.named_children()):
        if isinstance(child, nn.Linear) and child_name in TARGETS:
            setattr(mod, child_name, LoRALinear(child, RANK, ALPHA).to(dev)); n_lora += 1
params = [q for q in model.parameters() if q.requires_grad]
print('lora layers', n_lora, 'trainable params %.1fM' % (sum(q.numel() for q in params)/1e6), flush=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.config.use_cache = False

def lm_loss(ids, att, lab):
    # logits only at note positions (full-vocab logits over every position would dominate memory)
    h = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    sel = lab[:, 1:] != -100
    logits = model.lm_head(h[:, :-1][sel]).float()
    return F.cross_entropy(logits, lab[:, 1:][sel])

# ----------------------------------------------------------------------------------------------
# Training (fixed plan)
# ----------------------------------------------------------------------------------------------
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.99))
steps_per_epoch = math.ceil(len(train_enc) / (BS * ACCUM))
total = steps_per_epoch * EPOCHS; warm = max(1, int(0.05 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/warm) * 0.5 * (1 + math.cos(math.pi * min(1, s/total))))
print('optimiser steps', total, flush=True)

def batches(epoch):
    rng = random.Random(SEED + epoch)
    idx = list(range(len(train_enc))); rng.shuffle(idx)
    chunk = BS * 64; out = []
    for s in range(0, len(idx), chunk):              # length-bucketing inside shuffled chunks
        c = sorted(idx[s:s+chunk], key=lambda i: len(train_enc[i][0]))
        out += [c[i:i+BS] for i in range(0, len(c), BS)]
    rng.shuffle(out)
    return out

def collate(b):
    L = max(len(train_enc[i][0]) for i in b)
    ids = torch.full((len(b), L), tok.pad_token_id); lab = torch.full((len(b), L), -100); att = torch.zeros((len(b), L), dtype=torch.long)
    for j, i in enumerate(b):
        x, y = train_enc[i]; ids[j, :len(x)] = torch.tensor(x); lab[j, :len(y)] = torch.tensor(y); att[j, :len(x)] = 1
    return ids.to(dev), att.to(dev), lab.to(dev)

model.train(); step = 0; t0 = time.time()
for ep in range(EPOCHS):
    bl = batches(ep); run_loss = 0.0; nb = 0
    for bi, b in enumerate(bl):
        ids, att, lab = collate(b)
        loss = lm_loss(ids, att, lab)
        (loss / ACCUM).backward(); run_loss += loss.item(); nb += 1
        if (bi + 1) % ACCUM == 0 or bi == len(bl) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 50 == 0:
                print(f'ep {ep} step {step}/{total} loss {run_loss/nb:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s', flush=True)
                run_loss = 0.0; nb = 0
print('training done %.0fs' % (time.time()-t0), flush=True)

# ----------------------------------------------------------------------------------------------
# Decoding: beam + samples, MBR selection with chrF utility (each test row on its own)
# ----------------------------------------------------------------------------------------------
model.eval(); model.gradient_checkpointing_disable(); model.config.use_cache = True

def collate_prompts(ctxs):
    enc = [tok(prompt(c)).input_ids[-(MAXLEN - 64):] for c in ctxs]
    L = max(len(e) for e in enc)
    ids = torch.full((len(enc), L), tok.pad_token_id); att = torch.zeros((len(enc), L), dtype=torch.long)
    for i, e in enumerate(enc):
        ids[i, L-len(e):] = torch.tensor(e); att[i, L-len(e):] = 1
    return ids.to(dev), att.to(dev)

def clean(s):
    s = s.split('\n')[0].strip()
    return s if s else 'Summa.'

@torch.no_grad()
def generate(df):
    order = np.argsort([len(c) for c in df.context])[::-1]
    beam_out = [None]*len(df); samp_out = [None]*len(df)
    for s in range(0, len(df), GEN_BS):
        idx = order[s:s+GEN_BS]
        ids, att = collate_prompts([df.context.iloc[i] for i in idx])
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=56, num_beams=BEAMS, do_sample=False,
                           pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for i, t in zip(idx, tok.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)):
            beam_out[i] = clean(t)
        sb = max(1, GEN_BS // (NSAMP // 4))
        for s2 in range(0, len(idx), sb):
            idx2 = idx[s2:s2+sb]
            ids2, att2 = collate_prompts([df.context.iloc[i] for i in idx2])
            g = model.generate(input_ids=ids2, attention_mask=att2, max_new_tokens=56, do_sample=True, temperature=TEMP, top_p=0.95,
                               num_return_sequences=NSAMP, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
            txt = tok.batch_decode(g[:, ids2.shape[1]:], skip_special_tokens=True)
            for j, i in enumerate(idx2):
                samp_out[i] = [clean(t) for t in txt[j*NSAMP:(j+1)*NSAMP]]
        if (s // GEN_BS) % 10 == 0:
            print(f'  gen {s}/{len(df)} {time.time()-t0:.0f}s', flush=True)
    return beam_out, samp_out

def mbr_select(beam, samples):
    uniq = list(dict.fromkeys([beam] + samples))
    best, bu = None, -1.0
    for c in uniq:
        u = sum(chrf(c, s) for s in samples) / len(samples)
        if u > bu:
            best, bu = c, u
    return best

beam, samp = generate(te)
notes = [restore(mbr_select(b, s), c) for b, s, c in zip(beam, samp, te.context)]

# ----------------------------------------------------------------------------------------------
# Submission (validated: non-empty, single line, <= 200 chars)
# ----------------------------------------------------------------------------------------------
def fix(n):
    n = ' '.join(str(n).split())[:200]
    return n if n else 'Summa.'
sub = pd.DataFrame({'example_id': te.example_id, 'note': [fix(n) for n in notes]})
assert len(sub) == len(te) and sub.example_id.is_unique and (sub.note.str.len() > 0).all()
os.makedirs(os.path.dirname(os.path.abspath(SUB_PATH)), exist_ok=True)
sub.to_csv(SUB_PATH, index=False)
print('wrote', SUB_PATH, len(sub), 'rows; total time %.0fs' % (time.time()-t0), flush=True)
