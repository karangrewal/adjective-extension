""" utility functions used by all predictive models """

import numpy as np
import pickle
from scipy.sparse import load_npz
from scipy.spatial.distance import jensenshannon


def configure_output_file(out_path, model_name, dataset_name, metric_name):
    out_info = '\n{}\n'.format('-' * 80)
    out_info += '> job for {} model, started {}\n\n'.format(model_name, time.ctime())
    out_info += '> evaluation metric: {}\n'.format(metric_name)
    out_info += '> adjective set: {}\n'.format(dataset_name)
    out_info += '>* consider all future pairings\n\n' if ALL_FUTURE else '\n'
    open(out_path, 'w').write(out_info)


def empirical_distribution(t_new):
    """ return the empirical distribution at time t + delta """
    if ALL_FUTURE:
        p_empirical = load_cooc(t_new, 2001)
        p_empirical = 1. * (p_empirical >= THRES) * p_empirical
        p_empirical = p_empirical.sum(axis=0)
    else:
        p_empirical = load_cooc(t_new, t_new)
        p_empirical = 1. * (p_empirical >= THRES) * p_empirical
    return p_empirical


def get_class_mask(t):
    return 1. * (load_cooc(t, t) >= THRES)


def get_cooc_mask(t):
    cooc = 1. * (load_cooc(1800, t) >= THRES)
    cooc = 1. * np.any(cooc, axis=0)
    return cooc


def get_ground_truth(t):
    if ALL_FUTURE:
        novel_cooc = 1. * (load_cooc(t + 10, 2001) >= THRES) 
        novel_cooc = 1. * np.any(novel_cooc, axis=0) * (1. - get_cooc_mask())
    else:
        novel_cooc = 1. * (load_cooc(t + 10, t + 10) >= THRES) * (1. - get_cooc_mask())
    ground_truth = np.tile(np.arange(-1, -novel_cooc.shape[1] - 1, -1), (novel_cooc.shape[0], 1))
    assert ground_truth.shape == novel_cooc.shape
    novel_cooc = np.argwhere(novel_cooc == 1.)
    for i in range(ground_truth.shape[0]):
        place_index = 0
        for x, y in novel_cooc:
            if x == i:
                ground_truth[i, place_index] = y
                place_index += 1
    return ground_truth


def load_adjectives(adjective_set_name):
	if adjective_set_name not in ('frq200', 'rand200', 'syn65'):
		print('error: invalid adjective set specified')
		return None
	return pickle.load(open('../data/{}/{}.pkl'.format(adjective_set_name, adjective_set_name)))


def load_cooc(t_start, t_end):
    """ return cooccurrence matrix from t_start to t_end (inclusive) """
    cooc = [load_npz('../data/historical/cooc_{}.npz'.format(r)) for r in range(t_start, t_end+1, 10)]
    cooc = [x[get_noun_indices_t(t_start), :] for x in cooc]
    cooc = [x[:, get_adj_indices()] for x in cooc]
    cooc = np.array([x.tocoo().toarray() for x in cooc])
    assert cooc.ndim == 3
    if cooc.shape[0] == 1:
        cooc = cooc[0]
    return cooc


def jsd(P, Q):
    assert P.shape == Q.shape
    divs = list()
    for i in range(P.shape[0]):
        if P[i,:].sum() > 0. and Q[i,:].sum() > 0.:
            divs.append(jensenshannon(P[i,:], Q[i,:]))
    return np.array(divs)


def load_year(year):
    """ load vocab and pre-trained word vectors for specified year """
    def to_strs(V):
        strs = list()
        for word in V:
            try:
                strs.append(word.__str__())
            except UnicodeEncodeError:
                strs.append('')
        return strs
    vocab = pickle.load(open('../data/diachronic_embeddings/{}-vocab.pkl'.format(year), 'rb'))
    embeddings = np.load('../data/diachronic_embeddings/{}-w.npy'.format(year))
    return to_strs(vocab), embeddings


def prior(t, normalized=True):
    """ returns array with shape (n_adjectives,) where entry j is the 
    normalized/unnormalized prior probability of adjective j """
    p = load_cooc(t, t)
    p = 1. * (p >= THRES)
    p = np.sum(p, axis=0)
    return p / np.sum(p) if normalized else p


###################### FOR INDEXING CO-OCCURRENCE MATRICES #####################

def get_noun_indices_t(t):
    """ return numpy array of indices of nouns to consider at time t """
    nouns_dataset = open('../data/{}/nouns_{}.txt'.format(DATASET, t)).read().strip().split('\n')
    noun2ind = pickle.load(open('../data/wordnet/nouns_wn.pkl', 'rb'))
    inds = list()
    for n in nouns_dataset:
        if n in V_t and not all(W_t[V_t.index(n)]==0.):
            inds.append( noun2ind[n] )
    return inds


def get_adj_indices():
    """ return numpy array of indices of adjectives """
    adj2ind = pickle.load(open('../data/wordnet/adjectives_wn.pkl', 'rb'))
    return np.array([adj2ind[a] for a in adjectives])
