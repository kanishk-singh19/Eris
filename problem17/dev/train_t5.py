#!/usr/bin/env python3
"""ByT5 (seq2seq, byte-level) + LoRA for marginal-note generation; author-grouped validation."""
import os, sys, math, time, json, argparse, random
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import canon, restore, author_folds, ANCHOR, crop_context
from chrf import chrf, mean_chrf

p = argparse.ArgumentParser()
p.add_argument('--model', default='google/byt5-small')
p.add_argument('--fold', type=int, default=0)
p.add_argument('--epochs', type=float, default=2)
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--rank', type=int, default=32)
p.add_argument('--alpha', type=int, default=64)
p.add_argument('--bs', type=int, default=4)
p.add_argument('--accum', type=int, default=4)
p.add_argument('--before', type=int, default=450)   # char window around the anchor
p.add_argument('--after', type=int, default=520)
p.add_argument('--maxlen', type=int, default=1024)
p.add_argument('--gc', type=int, default=1)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default='t5run')
p.add_argument('--nsamp', type=int, default=16)
p.add_argument('--beams', type=int, default=4)
p.add_argument('--temp', type=float, default=0.8)
p.add_argument('--gen_bs', type=int, default=16)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--eval_every', type=int, default=0)
p.add_argument('--val_limit', type=int, default=0)
p.add_argument('--predict_test', type=int, default=0)
args = p.parse_args()
print(vars(args), flush=True)
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
dev = 'cuda'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained(args.model)

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
tr = pd.read_csv(os.path.join(D, 'train.csv')); te = pd.read_csv(os.path.join(D, 'test.csv'))
tr['fold'] = author_folds(tr)
if args.fold >= 0:
    trn, val = tr[tr.fold != args.fold].reset_index(drop=True), tr[tr.fold == args.fold].reset_index(drop=True)
else:
    trn, val = tr, te
if args.limit: trn = trn.sample(args.limit, random_state=0).reset_index(drop=True)
if args.val_limit and args.fold >= 0: val = val.sample(min(args.val_limit, len(val)), random_state=0).reset_index(drop=True)
print('train', len(trn), 'eval', len(val), flush=True)

def src(ctx):
    return canon(crop_context(ctx, args.before, args.after))
def enc_src(ctx):
    return tok(src(ctx)).input_ids[:args.maxlen]
train_enc = [(enc_src(c), tok(canon(n)).input_ids[:128]) for c, n in zip(trn.context, trn.note)]
print('max src len', max(len(x[0]) for x in train_enc), 'max tgt', max(len(x[1]) for x in train_enc), flush=True)

model = AutoModelForSeq2SeqLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
for q in model.parameters(): q.requires_grad_(False)

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.05):
        super().__init__()
        self.base = base; self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5)); self.drop = nn.Dropout(dropout)
    def forward(self, x):
        y = self.base(x)
        if self.training:
            z = F.linear(F.linear(self.drop(x).float(), self.A), self.B) * self.scale
        else:
            z = F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale
        return y + z.to(y.dtype)

TARGETS = ('q', 'k', 'v', 'o', 'wi_0', 'wi_1', 'wo')
n = 0
for name, mod in list(model.named_modules()):
    for cn, ch in list(mod.named_children()):
        if isinstance(ch, nn.Linear) and cn in TARGETS:
            setattr(mod, cn, LoRALinear(ch, args.rank, args.alpha).to(dev)); n += 1
params = [q for q in model.parameters() if q.requires_grad]
print('lora layers', n, 'trainable %.1fM' % (sum(q.numel() for q in params)/1e6), flush=True)
if args.gc:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.config.use_cache = False

def collate_src(ctxs):
    enc = [enc_src(c) for c in ctxs]; L = max(len(e) for e in enc)
    ids = torch.full((len(enc), L), tok.pad_token_id); att = torch.zeros((len(enc), L), dtype=torch.long)
    for i, e in enumerate(enc): ids[i, :len(e)] = torch.tensor(e); att[i, :len(e)] = 1
    return ids.to(dev), att.to(dev)

def clean(s):
    s = s.split('\n')[0].strip(); return s if s else 'Summa.'

@torch.no_grad()
def generate(df, nsamp, beams, temp):
    model.eval(); model.config.use_cache = True
    was_gc = model.is_gradient_checkpointing
    if was_gc: model.gradient_checkpointing_disable()
    order = np.argsort([len(c) for c in df.context])[::-1]
    out_beam = [None]*len(df); out_samp = [None]*len(df); bs = args.gen_bs
    for s in range(0, len(df), bs):
        idx = order[s:s+bs]
        ids, att = collate_src([df.context.iloc[i] for i in idx])
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=128, num_beams=beams, do_sample=False)
        for i, t in zip(idx, tok.batch_decode(g, skip_special_tokens=True)): out_beam[i] = clean(t)
        if nsamp:
            sb = max(1, bs // max(1, nsamp // 4))
            for s2 in range(0, len(idx), sb):
                idx2 = idx[s2:s2+sb]; ids2, att2 = collate_src([df.context.iloc[i] for i in idx2])
                g = model.generate(input_ids=ids2, attention_mask=att2, max_new_tokens=128, do_sample=True, temperature=temp, top_p=0.95, num_return_sequences=nsamp)
                txt = tok.batch_decode(g, skip_special_tokens=True)
                for j, i in enumerate(idx2): out_samp[i] = [clean(t) for t in txt[j*nsamp:(j+1)*nsamp]]
        if s % (bs*10) == 0: print(f'  gen {s}/{len(df)}', flush=True)
    if was_gc: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.config.use_cache = False; model.train()
    return out_beam, out_samp

def mbr(cands_list, samples_list):
    out = []
    for cands, samples in zip(cands_list, samples_list):
        uniq = list(dict.fromkeys(cands + samples)); best, bu = None, -1
        for c in uniq:
            u = sum(chrf(c, s) for s in samples) / len(samples)
            if u > bu: best, bu = c, u
        out.append(best)
    return out

def evaluate(df, tag):
    t0 = time.time(); beam, samp = generate(df, args.nsamp, args.beams, args.temp)
    res = {'beam': [restore(b, c) for b, c in zip(beam, df.context)]}
    if args.nsamp:
        res['mbr'] = [restore(m, c) for m, c in zip(mbr([[b] for b in beam], samp), df.context)]
    if 'note' in df:
        for k, v in res.items(): print(f'[{tag}] chrF {k}: {mean_chrf(v, df.note):.4f}', flush=True)
    print(f'  gen time {time.time()-t0:.0f}s', flush=True)
    return res, samp

opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
steps_per_epoch = math.ceil(len(train_enc) / (args.bs * args.accum))
total = int(steps_per_epoch * args.epochs); warm = max(1, int(0.05 * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/warm) * 0.5 * (1 + math.cos(math.pi * min(1, s/total))))
print('total opt steps', total, flush=True)

def batches(epoch):
    rng = random.Random(args.seed + epoch); idx = list(range(len(train_enc))); rng.shuffle(idx)
    chunk = args.bs * 64; out = []
    for s in range(0, len(idx), chunk):
        c = sorted(idx[s:s+chunk], key=lambda i: len(train_enc[i][0])); out += [c[i:i+args.bs] for i in range(0, len(c), args.bs)]
    rng.shuffle(out); return out

def collate(b):
    Ls = max(len(train_enc[i][0]) for i in b); Lt = max(len(train_enc[i][1]) for i in b)
    ids = torch.full((len(b), Ls), tok.pad_token_id); att = torch.zeros((len(b), Ls), dtype=torch.long); lab = torch.full((len(b), Lt), -100)
    for j, i in enumerate(b):
        x, y = train_enc[i]; ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1; lab[j, :len(y)] = torch.tensor(y)
    return ids.to(dev), att.to(dev), lab.to(dev)

model.train(); step = 0; t0 = time.time(); done = False; ep = 0
while not done:
    bl = batches(ep); run_loss = 0; nb = 0
    for bi, b in enumerate(bl):
        ids, att, lab = collate(b)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        (loss / args.accum).backward(); run_loss += loss.item(); nb += 1
        if (bi + 1) % args.accum == 0 or bi == len(bl) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 50 == 0:
                print(f'ep {ep} step {step}/{total} loss {run_loss/nb:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.2f}G', flush=True)
                run_loss = 0; nb = 0
            if args.eval_every and step % args.eval_every == 0 and 'note' in val:
                ns = args.nsamp; args.nsamp = 0; evaluate(val.sample(min(300, len(val)), random_state=0), f'step{step}'); args.nsamp = ns
            if step >= total: done = True; break
    ep += 1

res, samp = evaluate(val, 'final')
outp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', f'{args.out}_f{args.fold}.json')
json.dump({'ids': list(val.example_id), 'res': res, 'samples': samp, 'args': vars(args)}, open(outp, 'w'), ensure_ascii=False)
print('saved', outp, flush=True)
if args.predict_test and args.fold >= 0:
    res_te, samp_te = evaluate(te, 'test')
    json.dump({'ids': list(te.example_id), 'res': res_te, 'samples': samp_te, 'args': vars(args)}, open(outp.replace('.json', '_test.json'), 'w'), ensure_ascii=False)
    print('saved test preds', flush=True)
