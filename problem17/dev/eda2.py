import pandas as pd, numpy as np, re, unicodedata, sys
sys.path.insert(0, 'dev')
from chrf import chrf, norm, mean_chrf
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
COMB = 'ͤ'
for name, df in [('train', tr), ('test', te)]:
    c = df.context
    print(name, 'ctx has ſ', c.str.contains('ſ').mean(), 'ctx has comb-e', c.str.contains(COMB).mean(),
          'ctx has umlaut', c.str.contains('[äöüÄÖÜ]').mean(), 'ctx has ꝛ', c.str.contains('ꝛ').mean(),
          'ctx has ẽ/ñ/ũ tilde', c.str.contains('[ẽñũãõ]').mean())
n = tr.note
print('note has ſ', n.str.contains('ſ').mean(), '| given ctx ſ:', n[tr.context.str.contains('ſ')].str.contains('ſ').mean(),
      '| given no ctx ſ:', n[~tr.context.str.contains('ſ')].str.contains('ſ').mean())
has_c = tr.context.str.contains(COMB); has_u = tr.context.str.contains('[äöüÄÖÜ]')
print('ctx comb only', (has_c & ~has_u).mean(), 'umlaut only', (~has_c & has_u).mean(), 'both', (has_c & has_u).mean(), 'neither', (~has_c & ~has_u).mean())
print('note comb | ctx comb only', n[has_c & ~has_u].str.contains(COMB).mean(), ' note umlaut | ctx comb only', n[has_c & ~has_u].str.contains('[äöüÄÖÜ]').mean())
print('note umlaut | ctx umlaut only', n[~has_c & has_u].str.contains('[äöüÄÖÜ]').mean(), ' note comb | ctx umlaut only', n[~has_c & has_u].str.contains(COMB).mean())
# NFKC effects
print('NFKC ſ ->', unicodedata.normalize('NFKC','ſ'), ' oͤ ->', repr(unicodedata.normalize('NFKC','oͤ')), ' ꝛ ->', unicodedata.normalize('NFKC','ꝛ'), ' ﬀ->', unicodedata.normalize('NFKC','ﬀ'), ' ñ ->', repr(unicodedata.normalize('NFKC','ñ')))
# character inventory of notes after norm
from collections import Counter
cc = Counter(ch for s in n for ch in norm(s))
print('note char inventory (normalised) top 80:', cc.most_common(80))
rare = [(k,v) for k,v in cc.items() if not k.isascii()]
print('non-ascii normalised note chars:', sorted(rare, key=lambda x:-x[1])[:60])
# where do note words come from relative to anchor
def wordpos(row):
    ctx = unicodedata.normalize('NFKC', row.context).lower(); a = ctx.find('⟦note⟧')
    out = []
    for w in unicodedata.normalize('NFKC', row.note).lower().split():
        w = w.strip('.,;:/()')
        if len(w) < 4: continue
        i = ctx.find(w)
        if i >= 0: out.append(i - a)
    return out
pos = [p for r in tr.itertuples() for p in wordpos(r)]
pos = np.array(pos); print('word offsets rel. to anchor: n', len(pos), 'pct', np.percentile(pos,[5,10,25,50,75,90,95]))
print('frac within [-200, 400]', ((pos>-200)&(pos<400)).mean(), ' within [-100,300]', ((pos>-100)&(pos<300)).mean(), 'before anchor', (pos<0).mean())
# per-author note style consistency: digit frac per author
g = tr.groupby('author_id').note.apply(lambda s: s.str.contains(r'\d').mean())
print('per-author digit frac distribution', g.describe())
# work-level patterns? roman numerals
print('roman only notes', n.str.match(r'^[IVXLC]+\.?$').mean())
# first char uppercase
print('ends with period', n.str.endswith('.').mean())
# what precedes the anchor: is it at a sentence start?
print('anchor at ctx start', (tr.context.str.startswith('⟦NOTE⟧')).mean(), (te.context.str.startswith('⟦NOTE⟧')).mean())
