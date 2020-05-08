## 1. Adjective-noun cooccurrence data

The `data/` folder contains all historical adjective-noun cooccurrence counts grouped by decade, as well as adjectives and nouns used in our experiments. The following python code (executed from the `scripts/` folder) accesses the 

```
from util import *

# load the adjective set FRQ-200
adjectives = load_adjectives('frq200')

# retrieve the set of nouns to consider in the 1970s given our choice of 
# adjective set
noun2ind = pickle.load(open('../data/wordnet/nouns_wn.pkl', 'rb'))
ind2noun = {i: a for a, i in noun2ind.iteritems()}
nouns_t_i = get_noun_indices_t(t)
nouns_t = [ind2noun[i] for i in nouns_t_i]

# C is a 3D numpy array where the (t, i, j) entry gives the raw cooccurrence 
# count between noun i and adjective j in decade t (offset by 1970). 
C = load_cooc(1970, 1990)

# print the raw co-occurrence count between "computer" and "artificial" in the 
# 1980s
i, j = nouns_t.index('computer'), adjectives.index('artificial')
print(C[1, i, j])
```

## 2. Predictive models

THe `scripts/` folder contains the exemplar, prototype, progenitor, k-nearest neighbors, and baseline models. One of these models can be executed as

```bash
$ scripts/exemplar.py --frq200 --precision
```

where `--frq200` specified the "FRQ-200" adjective set, and `--precision` specifies we want to evaluate models based on predictive accuracy. Instead, you can use `--jsd` to specify the Jensen-Shannon divergence between the predicted and empirical posterior distributions over adjectives. In addition, the options `--future` and `--map` can be used to consider all future pairings when evaluating models and to use kernel parameters learned via MAP estimates, respectively.

Note that the prototype and progenitor models are both wrapped into `scripts/prototype.py`.