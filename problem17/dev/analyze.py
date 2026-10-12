#!/usr/bin/env python3
"""Per-category chrF breakdown of a saved run (dev/out/<name>_f<k>.json)."""
import sys, json, os, re, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chrf import chrf
tr = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'train.csv')).set_index('example_id')
d = json.load(open(sys.argv[1])); key = sys.argv[2] if len(sys.argv) > 2 else 'mbr'
hyp = d['res'][key]; ids = d['ids']; refs = [tr.note[i] for i in ids]
def cat(n):
    if re.search(r'\d', n): return 'ref' if re.search(r'^[A-Za-z]{1,6}\.?\s*\d', n) or re.search(r'\d+\s*,\s*\d+', n) else 'num'
    return 'short' if len(n) <= 12 else 'topic'
df = pd.DataFrame({'id': ids, 'ref': refs, 'hyp': hyp, 'author': [tr.author_id[i] for i in ids]})
df['sc'] = [chrf(h, r) for h, r in zip(df.hyp, df.ref)]; df['cat'] = df.ref.map(cat)
print('overall', df.sc.mean().round(4)); print(df.groupby('cat').sc.agg(['mean', 'size']).round(3))
print(df.groupby('author').sc.agg(['mean', 'size']).round(3).sort_values('mean'))
print('len hyp/ref', df.hyp.str.len().mean().round(1), df.ref.str.len().mean().round(1))
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 70)
print(df.sample(30, random_state=1)[['ref', 'hyp', 'sc']].to_string())
