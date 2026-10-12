import unicodedata, re, numpy as np, pandas as pd

COMB = 'ͤ'
UML = {'a'+COMB: 'ä', 'o'+COMB: 'ö', 'u'+COMB: 'ü', 'A'+COMB: 'Ä', 'O'+COMB: 'Ö', 'U'+COMB: 'Ü'}
UML_BACK = {v: k for k, v in UML.items()}
ANCHOR = '⟦NOTE⟧'

def canon(s):
    """Model-side spelling: NFKC (long s -> s, ligatures expanded; the scorer does the same),
    and the historical 'e above' combining mark folded into modern umlauts."""
    s = unicodedata.normalize('NFKC', str(s))
    for k, v in UML.items():
        s = s.replace(k, v)
    return s

def convention(ctx):
    """Which umlaut convention does THIS passage use (row-local, no other rows involved)."""
    has_comb = COMB in ctx
    has_uml = re.search('[äöüÄÖÜ]', ctx) is not None
    if has_comb and not has_uml:
        return 'comb'
    return 'uml'

def restore(note, ctx):
    """Map the model's modern umlauts back to the passage's own convention."""
    if convention(ctx) == 'comb':
        for k, v in UML_BACK.items():
            note = note.replace(k, v)
    return note

def author_folds(df, nfold=5, seed=0):
    """Group K-fold by author (test authors are unseen); balanced by row count."""
    rng = np.random.RandomState(seed)
    authors = df.groupby('author_id').size().sort_values(ascending=False)
    fold_of = {}; load = [0]*nfold
    order = list(authors.index)
    # greedy balanced assignment with a little randomness among the smallest-load folds
    for a in order:
        k = int(np.argmin(load)); fold_of[a] = k; load[k] += authors[a]
    return df.author_id.map(fold_of).values

def crop_context(ctx, before=800, after=900):
    """Keep a char window around the anchor (row-local)."""
    a = ctx.find(ANCHOR)
    s = max(0, a - before); e = min(len(ctx), a + len(ANCHOR) + after)
    return ctx[s:e]
