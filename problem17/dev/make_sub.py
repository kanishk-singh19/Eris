#!/usr/bin/env python3
"""Build submission.csv from saved test predictions (one or more *_test.json files; candidates pooled).
usage: make_sub.py out.csv [--concat K --topk T] file1_test.json [file2_test.json ...]"""
import sys, json, os, itertools, argparse, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chrf import chrf
from common import restore
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('files', nargs='+')
ap.add_argument('--concat', type=int, default=0); ap.add_argument('--topk', type=int, default=3)
a = ap.parse_args()
te = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'test.csv')).set_index('example_id')
ids = None; B = None; S = None
for f in a.files:
    d = json.load(open(f))
    if ids is None: ids = d['ids']; B = [[] for _ in ids]; S = [[] for _ in ids]
    assert ids == d['ids']
    for i, (b, s) in enumerate(zip(d['res']['beam'], d['samples'])): B[i].append(b); S[i] += s
def util(c, samp): return sum(chrf(c, s) for s in samp) / len(samp)
notes = []
for i, ex in enumerate(ids):
    samp = S[i]; uniq = list(dict.fromkeys(B[i] + samp))
    scored = sorted(((util(c, samp), c) for c in uniq), reverse=True); bu, best = scored[0]
    if a.concat:
        top = [c for _, c in scored[:a.topk]]
        for k in range(2, a.concat + 1):
            for combo in itertools.permutations(top, k):
                c = ' '.join(combo)
                if len(c) > 200: continue
                u = util(c, samp)
                if u > bu: bu, best = u, c
    n = ' '.join(restore(best, te.context[ex]).split())[:200]
    notes.append(n if n else 'Summa.')
sub = pd.DataFrame({'example_id': ids, 'note': notes})
sample = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'sample_submission.csv'))
sub = sub.set_index('example_id').loc[sample.example_id].reset_index()
assert list(sub.columns) == ['example_id', 'note'] and len(sub) == len(sample) and sub.example_id.is_unique
assert (sub.note.str.len() > 0).all() and (sub.note.str.len() <= 200).all() and not sub.note.str.contains('\n').any()
sub.to_csv(a.out, index=False); print('wrote', a.out, len(sub), 'mean len', sub.note.str.len().mean().round(1))
print(sub.head(10).to_string())
