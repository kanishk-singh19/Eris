import unicodedata
from collections import Counter

def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    return ''.join(s.split())

def _ngrams(s, n):
    return Counter(s[i:i+n] for i in range(len(s)-n+1))

def chrf(hyp, ref, N=6, beta=2.0):
    """Sentence chrF (Popovic 2015) on NFKC-lowercased whitespace-stripped strings."""
    h, r = norm(hyp), norm(ref)
    if not h or not r:
        return 0.0
    precs, recs = [], []
    for n in range(1, N+1):
        hg, rg = _ngrams(h, n), _ngrams(r, n)
        th, tr = sum(hg.values()), sum(rg.values())
        if th == 0 or tr == 0:
            # sacrebleu-style: orders with no n-grams in either are skipped via effective order
            continue
        m = sum((hg & rg).values())
        precs.append(m/th); recs.append(m/tr)
    if not precs:
        return 0.0
    P = sum(precs)/len(precs); R = sum(recs)/len(recs)
    if P + R == 0:
        return 0.0
    b2 = beta*beta
    return (1+b2)*P*R/(b2*P+R)

def mean_chrf(hyps, refs):
    return sum(chrf(h, r) for h, r in zip(hyps, refs))/len(refs)
