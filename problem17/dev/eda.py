import pandas as pd, numpy as np, re, unicodedata, sys
sys.path.insert(0, 'dev')
from chrf import chrf, norm, mean_chrf
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
print(tr.shape, te.shape)
print('authors train', tr.author_id.nunique(), 'test', te.author_id.nunique())
print(tr.groupby('author_id').size().describe())
print(te.groupby('author_id').size().describe())
tr['nlen'] = tr.note.str.len(); print('note len', tr.nlen.describe())
tr['clen'] = tr.context.str.len(); print('ctx len', tr.clen.describe())
print('test ctx len', te.context.str.len().describe())
# anchor position
def pos(c):
    i = c.find('⟦NOTE⟧'); return i/len(c)
print('anchor rel pos train', tr.context.map(pos).describe())
print('anchor rel pos test', te.context.map(pos).describe())
print('all have anchor', tr.context.str.contains('⟦NOTE⟧').all(), te.context.str.contains('⟦NOTE⟧').all())
# digits
print('digit frac', tr.note.str.contains(r'\d').mean())
# copy rate: fraction of note chars (normalised) found as substring of context
def copy_frac(row):
    n = norm(row.note); c = norm(row.context.replace('⟦NOTE⟧',''))
    # longest common substring coverage approx: greedy token-based
    toks = unicodedata.normalize('NFKC', row.note).lower().split()
    ctok = unicodedata.normalize('NFKC', row.context).lower()
    hit = sum(len(t) for t in toks if t.strip('.,;:') and t.strip('.,;:') in ctok)
    tot = sum(len(t) for t in toks)
    return hit/max(tot,1)
tr['copy'] = tr.apply(copy_frac, axis=1)
print('copy frac', tr['copy'].describe())
print('frac notes fully copyable (>=0.99)', (tr['copy']>=0.99).mean(), ' >=0.5', (tr['copy']>=0.5).mean())
# whole note as substring
tr['sub'] = tr.apply(lambda r: norm(r.note.rstrip('.')) in norm(r.context), axis=1)
print('note substring of ctx', tr['sub'].mean())
# trivial baselines
print('Summa.', mean_chrf(['Summa.']*len(tr), tr.note))
for k in [10,15,20,25,30,40]:
    hyp = tr.context.map(lambda c: c.split('⟦NOTE⟧')[1][:k])
    print('after-anchor first', k, round(mean_chrf(hyp, tr.note),4))
for k in [15,20,30]:
    hyp = tr.context.map(lambda c: c.split('⟦NOTE⟧')[0][-k:])
    print('before-anchor last', k, round(mean_chrf(hyp, tr.note),4))
# type categories
def cat(n):
    if re.search(r'\d', n):
        if re.search(r'^[A-Za-z]{1,6}\.?\s*\d', n) or re.search(r'\d+\s*,\s*\d+', n): return 'ref'
        return 'num-other'
    if len(n) <= 12: return 'short'
    return 'topic'
tr['cat'] = tr.note.map(cat); print(tr.cat.value_counts(normalize=True))
print(tr.sample(40, random_state=1)[['note','cat']].to_string())
# notes per author frequency reuse
print('unique notes', tr.note.nunique(), 'of', len(tr))
print(tr.note.value_counts().head(30))
