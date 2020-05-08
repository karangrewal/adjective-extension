""" An empirical evaluation of the type-based k-nearest neighbors model for 
adjective extension """

from util import *

import numpy as np
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import sys
import time

global DATASET, METRIC, ALL_FUTURE


def parse_command_line_args():
    global DATASET, METRIC, ALL_FUTURE
    DATASET = 'frq200' if '--frq200' in sys.argv else 'rand200' if '--rand200' in sys.argv else 'syn65' if '--syn65' in sys.argv else None
    METRIC = 'precision' if '--precision' in sys.argv else 'jsd' if '--jsd' in sys.argv else None
    
    # whether to consider all future pairings during evaluation
    ALL_FUTURE = True if '--future' in sys.argv else False

    if DATASET is None or METRIC is None:
        print('error: invalid or unspecified dataset or evaluation metric')
        exit()


if __name__ == '__main__':

    #################### THESE VARIABLES CAN BE MODIFIED #######################

    # compute posterior probabilities over adjectives for batches of nouns at a 
    # time to avoid memory errors, and this gives the batch size
    NOUN_PARTITION_SIZE = 2000

    # path to output files that score model performance and predictions
    METRIC_FILE = '../knn_metric.txt'
    PREDICTION_FILE = '../knn_predictions.txt'

    # the value at which to threshold adjective-noun co-occurrences in a decade
    THRES = 2

    ############################################################################

    parse_command_line_args()
    configure_output_file(METRIC_FILE, 'k-NN', DATASET, METRIC)
    adjectives = load_adjectives(DATASET)

    for t in range(1850, 1989, 10):
        V_t, W_t = load_year(t)

        ################### LOAD NOUNS AND THEIR EMBEDDINGS ####################

        noun2ind = pickle.load(open('../data/wordnet/nouns_wn.pkl', 'rb'))
        ind2noun = {i: a for a, i in noun2ind.iteritems()}
        nouns_t_i = get_noun_indices_t(t)
        nouns_t = [ind2noun[i] for i in nouns_t_i]
        noun_vectors = np.array([W_t[V_t.index(ind2noun[i])] for i in nouns_t_i])

        ########################################################################

        eligible_mask = 1. - get_cooc_mask(t)
        ground_truth = get_ground_truth(t)

        base = 0
        while base < len(nouns_t):
            end = min([base + NOUN_PARTITION_SIZE, len(nouns_t)])
            prediction_str = str()
            
            # rank-order each noun's closest neighbors
            closest = euclidean_distances(noun_vectors[base:end, :], noun_vectors)
            for i, noun in enumerate(nouns_t[base:end]):
                closest[i, base + i] = 999.
            closest = np.argsort(closest, axis=1)[:,:50]

            correct, jsd_scores = list(), list()
            for k in (1, 10):

                k_mask = closest[:,:k]
                k_mask = [(i, x) for i in range(k_mask.shape[0]) for x in k_mask[i,:]]
                _ = np.zeros((len(nouns_t[base:end]), len(nouns_t)))
                for (i, j) in k_mask:
                    _[i, j] = 1.

                k_mask = _[:,:].astype(np.float32)
                k_mask = np.tile(k_mask, (len(adjectives), 1, 1))
                k_mask = np.transpose(k_mask, axes=(1, 0, 2))

                posterior = load_cooc(1800, t).astype(np.float32).sum(axis=0)
                posterior = 1. * (posterior > 0.)
                posterior = np.tile(posterior, (end - base, 1, 1))
                posterior = np.transpose(posterior, axes=(0, 2, 1))
                posterior = posterior * k_mask

                posterior = np.sum(posterior, axis=2)
                posterior = posterior * eligible_mask[base:end, :]

                if METRIC == 'precision':

                    # break ties in k-NN predictions via bootstrap aggregating
                    correct_k_avg = list()
                    for n in range(50):

                        random_noise = np.random.uniform(high=0.5, size=posterior.shape) * eligible_mask[base:end, :]
                        posterior_n = posterior + random_noise
                        predictions = np.argsort(posterior_n, axis=1)[:,::-1]

                        correct_k, total = 0, 0
                        for i in range(predictions.shape[0]):
                    
                            total_i = int( np.sum(1. * (ground_truth[base + i, :] >= 0)) )
                            correct_i = np.intersect1d(predictions[i, :total_i], ground_truth[base + i, :total_i], assume_unique=True).shape[0]
                            
                            correct_k += correct_i
                            total += total_i

                            if n == 0 and total_i > 0:
                                prediction_str += '{},k={},{},{}\n'.format(t + 10, k, nouns_t[base + i], k, ','.join([adjectives[adj_ind] for adj_ind in predictions[i, :total_i]]))
                                prediction_str += '{},k={},{},{}\n\n'.format(t + 10, k, nouns_t[base + i], k, ','.join([adjectives[adj_ind] for adj_ind in ground_truth[base + i, :total_i]]))

                        correct_k_avg.append(correct_k)
                    correct.append(np.mean(correct_k_avg))

                elif METRIC == 'jsd':

                    p_empirical = empirical_distribution(t + 10)
                    p_empirical = p_empirical * (1. - get_cooc_mask(t))
                    
                    # don't need to break k-NN ties, unlike when evaluating 
                    # predictive accuracy
                    jsd_scores.append(jsd(posterior, p_empirical[base:end]).mean())
                    
                    # count of how many nouns we've made predictions for so far
                    n_nouns_predict = sum([1 if p_empirical[base + i,:].sum() > 0 else 0 for i in range(end - base)])
            
            if METRIC == 'precision':

                # report model predictions
                if DATASET == 'frq200' and METRIC == 'precision' and not ALL_FUTURE:
                    open(PREDICTION_FILE, 'a').write(prediction_str)
            
                # report predictive accuracy
                open(METRIC_FILE, 'a').write('{},{},{}\n'.format(t + 10, ','.join([str(x) for x in correct]), total))

            elif METRIC == 'jsd':

                open(METRIC_FILE, 'a').write('{},{},{}\n'.format(t + 10, ','.join([str(x) for x in jsd_scores]), n_nouns_predict))
            
            base += NOUN_PARTITION_SIZE
