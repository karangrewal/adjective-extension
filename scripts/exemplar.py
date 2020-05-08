""" An empirical evaluation of the type-based exemplar model for adjective 
extension """

from util import *

import numpy as np
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import sys
import time

global DATASET, METRIC, ALL_FUTURE, USE_MAP_ESTIMATE


def parse_command_line_args():
    global DATASET, METRIC, ALL_FUTURE, USE_MAP_ESTIMATE
    DATASET = 'frq200' if '--frq200' in sys.argv else 'rand200' if '--rand200' in sys.argv else 'syn65' if '--syn65' in sys.argv else None
    METRIC = 'precision' if '--precision' in sys.argv else 'jsd' if '--jsd' in sys.argv else None
    
    # whether to consider all future pairings during evaluation
    ALL_FUTURE = True if '--future' in sys.argv else False

    # whether to use MAP kernel values
    USE_MAP_ESTIMATE = True if '--map' in sys.argv else False

    if DATASET is None or METRIC is None or (METRCIC == 'jsd' and USE_MAP_ESTIMATE):
        print('error: invalid or unspecified dataset or evaluation metric')
        exit()


if __name__ == '__main__':

    #################### THESE VARIABLES CAN BE MODIFIED #######################

    # compute posterior probabilities over adjectives for batches of nouns at a 
    # time to avoid memory errors, and this gives the batch size
    NOUN_PARTITION_SIZE = 2000

    # path to output files that score model performance and predictions
    METRIC_FILE = '../exemplar_metric.txt'
    PREDICTION_FILE = '../exemplar_predictions.txt'

    # the value at which to threshold adjective-noun co-occurrences in a decade
    THRES = 2

    ############################################################################

    parse_command_line_args()
    configure_output_file(METRIC_FILE, 'exemplar', DATASET, METRIC)
    adjectives = load_adjectives(DATASET)

    # load kernel parameters
    kernel_params = pickle.load(open('../params/params.pkl', 'rb'))
    kernel_params = kernel_params['exemplar']['map' if USE_MAP_ESTIMATE else METRIC][DATASET]

    for t in range(1850, 1989, 10):
        V_t, W_t = load_year(t)

        ################### LOAD NOUNS AND THEIR EMBEDDINGS ####################

        noun2ind = pickle.load(open('../data/wordnet/nouns_wn.pkl', 'rb'))
        ind2noun = {i: a for a, i in noun2ind.iteritems()}
        nouns_t_i = get_noun_indices_t(t)
        nouns_t = [ind2noun[i] for i in nouns_t_i]
        noun_vectors = np.array([W_t[V_t.index(ind2noun[i])] for i in nouns_t_i])

        ########################################################################

        ground_truth = get_ground_truth(t)

        # normalizing term for exemplar
        exemplar_norm = get_class_mask(t).sum(axis=0)
        exemplar_norm = np.tile(exemplar_norm, (len(nouns_t), 1))
        np.place(exemplar_norm, exemplar_norm==0., 1.)

        base = 0
        while base < len(nouns_t):
            end = min([base + NOUN_PARTITION_SIZE, len(nouns_t)])
            prediction_str = str()

            probs = euclidean_distances(noun_vectors[base:end, :], noun_vectors)
            probs = np.exp(-(probs ** 2))
            probs = np.tile(probs, (len(adjectives), 1, 1))
            probs = np.transpose(probs, axes=(1, 0, 2))

            # apply 0/1 class mask to compute exemplars for each adjective
            probs = probs * np.tile(get_class_mask(t).T, (end - base, 1, 1))

            posterior = probs ** (1. / h_exemplar[t - 10])
            posterior = np.sum(posterior, axis=2)
            posterior = posterior * (1. / exemplar_norm[base:end, :])
            posterior = posterior * np.tile(prior(t), (end - base, 1))

            # apply 0/1 cooccurrence mask and rank-order adjectives by posterior
            posterior = posterior * (1. - get_cooc_mask(t)[base:end, :])

            if METRIC == 'precision':

                predictions = np.argsort(posterior, axis=1)[:,::-1]

                # compare predictions to ground truth
                correct, total = 0, 0
                for i in range(predictions.shape[0]):
                    
                    total_i = int( np.sum(1. * (ground_truth[base + i, :] >= 0)) )
                    correct_i = np.intersect1d(predictions[i, :total_i], ground_truth[base + i, :total_i], assume_unique=True).shape[0]
                    correct += correct_i
                    total += total_i

                    if total_i > 0:
                        prediction_str += '{},{},{}\n'.format(t + 10, nouns_t[base + i], ','.join([adjectives[adj_ind] for adj_ind in predictions[i, :total_i]]))
                        prediction_str += '{},{},{}\n\n'.format(t + 10, nouns_t[base + i], ','.join([adjectives[adj_ind] for adj_ind in ground_truth[base + i, :total_i]]))

                # report model predictions
                if DATASET == 'frq200' and METRIC == 'precision' not (ALL_FUTURE or USE_MAP_ESTIMATE):
                    open(PREDICTION_FILE, 'a').write(prediction_str)

                # report predictive accuracy
                open(METRIC_FILE, 'a').write('{},{},{}\n'.format(t + 10, correct, total))
            
            elif METRIC == 'jsd':

                p_empirical = empirical_distribution(t + 10)
                p_empirical = p_empirical * (1. - get_cooc_mask(t))
                
                jsd_score = jsd(posterior, p_empirical[base:end]).mean()

                # count of how many nouns we've made predictions for so far
                n_nouns_predict = sum([1 if p_empirical[base + i,:].sum() > 0 else 0 for i in range(end - base)])

                open(METRIC_FILE, 'a').write('{},{},{}\n'.format(t + 10, jsd_score, n_nouns_predict))

            base += NOUN_PARTITION_SIZE
