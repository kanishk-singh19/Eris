#!/usr/bin/env python3
"""Offline decoding study on saved samples: plain MBR vs MBR with concatenated candidates."""
import sys, json, os, itertools, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chrf import chrf, mean_chrf, norm
from common import restore
tr = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'train.csv')).set_index('example_id')
files = sys.argv[1:]
ids = None; beams = []; samples = []
for f in files:
    d = json.load(open(f))
    if ids is None: ids = d['ids']
    assert ids == d['ids']
    beams.append(d['res']['beam']); samples.append(d['samples'])
refs = [tr.note[i] for i in ids]; ctxs = [tr.context[i] for i in ids]
n = len(ids)
# merge candidates across files (ensembling by MBR)
B = [[b[i] for b in beams] for i in range(n)]
S = [sum((s[i] for s in samples), []) for i in range(n)]

def utility(c, samp):
    return sum(chrf(c, s) for s in samp) / len(samp)

def select(i, topk=3, concat=0, maxc=200):
    samp = S[i]; uniq = list(dict.fromkeys(B[i] + samp))
    scored = sorted(((utility(c, samp), c) for c in uniq), reverse=True)
    best_u, best = scored[0]
    if concat:
        top = [c for _, c in scored[:topk]]
        for k in range(2, concat + 1):
            for combo in itertools.permutations(top, k):
                c = ' '.join(combo)
                if len(c) > maxc: continue
                u = utility(c, samp)
                if u > best_u: best_u, best = u, c
    return best

for name, kw in [('beam0', None), ('mbr', dict(concat=0)), ('mbr+cat2', dict(topk=3, concat=2)), ('mbr+cat3', dict(topk=4, concat=3))]:
    if kw is None: hyp = [B[i][0] for i in range(n)]
    else: hyp = [select(i, **kw) for i in range(n)]
    hyp = [restore(h, c) for h, c in zip(hyp, ctxs)]
    sc = [chrf(h, r) for h, r in zip(hyp, refs)]
    print(f'{name:10s} chrF {np.mean(sc):.4f}  mean len {np.mean([len(h) for h in hyp]):.1f}', flush=True)
    if name == 'mbr+cat2':
        for i in range(8): print('   ', repr(refs[i]), '->', repr(hyp[i]), round(sc[i], 3))
