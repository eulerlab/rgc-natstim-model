import numpy as np
import pandas as pd
from scipy import interpolate
import pickle as pkl
from typing import Dict
from rgc_natstim_model.constants.context_change_detection import (
    DUR_MOVIE, RANDOM_SEQUENCES_PATH, GROUP_INFO_PATH, START_INDICES,
    NUM_CLIPS, NUM_CLIPS_TOTAL, NUM_VAL_CLIPS, CLIP_DUR, MOVIE_DUR
)

############################################# Response snippeting ######################################################


def get_resp_dicts(dataframe: pd.DataFrame,
                   reduce_fn=None,
                   normalize_fn=None):
    '''

    :param dataframe:
    :param reduce_fn:
    :param normalize_fn:
    :return:
    '''
    rnd = np.random.RandomState(seed=1000)
    val_clip_idx = list(rnd.choice(NUM_CLIPS, NUM_VAL_CLIPS, replace=False))
    train_clip_idx = list(np.arange(NUM_CLIPS))
    [train_clip_idx.remove(vci) for vci in val_clip_idx]

    full_movie_resp_dict = dataframe['responses_final']
    binned_movie_resp_dict = {}

    if reduce_fn is None:
        reduce_fn = lambda x, i: x[i:i + 30].mean() - x[i]
    else:
        raise NotImplementedError
    if normalize_fn is None:
        normalize_fn = lambda x: (x - x.mean())/x.std()
    else:
        raise NotImplementedError
    for nid, resp in full_movie_resp_dict.items():
        binned_movie_resp_dict[nid] = normalize_fn(
            np.asarray(
                [reduce_fn(resp, i) for i in START_INDICES]
            )
        )
    return full_movie_resp_dict, binned_movie_resp_dict


############################################# Movie snippeting ########################################################


def get_movie_contrast_by_session(session_ids, sess2movie_dict):
    movie_contrasts_session_ff = {session_id: np.zeros((2, len(START_INDICES)))
                                  for session_id in np.unique(session_ids)}

    for sid in np.unique(session_ids):
        temp_movie = sess2movie_dict[sid]
        movie_contrasts_session_ff[sid][0] = np.asarray(
            [np.mean(
                temp_movie[0, i, :30],
                axis=(-1, -2, -3)
            ) - np.mean(
                temp_movie[0, i - 1, -30:],
                axis=(-1, -2, -3)
            )
             for i in range(1, len(START_INDICES) + 1)])

        movie_contrasts_session_ff[sid][1] = np.asarray(
            [np.mean(
                temp_movie[1, i, :30],
                axis=(-1, -2, -3)
            ) - np.mean(
                temp_movie[1, i - 1, -30:],
                axis=(-1, -2, -3)
            )
             for i in range(1, len(START_INDICES) + 1)])
    return movie_contrasts_session_ff


class SessionMappings:
    def __init__(self, train_movie_snippets, test_movie_snippets,
                 random_sequences_path, group_info_path, dur_movie):
        self.sess2seq_test = {}
        self.sess2seqid = {}
        self.sess2seq = {}
        self.train_movie_snippets = train_movie_snippets
        self.test_movie_snippets = test_movie_snippets
        self.RANDOM_SEQUENCES_PATH = random_sequences_path
        self.GROUP_INFO_PATH = group_info_path
        self.DUR_MOVIE = dur_movie
        self.sess2vertical_transitions_dict = {}
        self.sess2brightness_transitions_dict = {}
        self.sess2movie_dict = {}
        self.sess2flatmovie_dict = {}
        self.sess2train_movie_dict = {}
        self.sess2flattrain_movie_dict = {}

    def load_sequences_info(self):
        with open(self.RANDOM_SEQUENCES_PATH, "rb") as f:
            self.ran_seq = np.load(f)

        with open(self.GROUP_INFO_PATH, "rb") as f:
            self.seq_info = pkl.load(f)

    def get_session_mappings(self, movie_data_dicts, dh_2_session_ids):
        self.load_sequences_info()

        for dh, session_ids in dh_2_session_ids.items():
            for session_id in session_ids:
                seq_id = movie_data_dicts[dh][session_id]['scan_sequence_idx']

                loc_labels = self.seq_info["visual_field_train"][self.ran_seq[:, seq_id]]
                lum_labels = self.seq_info["intensity_train"][self.ran_seq[:, seq_id]]

                loc_labels_full = np.concatenate([self.seq_info["visual_field_test"],
                                                  loc_labels[:54],
                                                  self.seq_info["visual_field_test"],
                                                  loc_labels[54:],
                                                  self.seq_info["visual_field_test"]])
                lum_labels_full = np.concatenate([self.seq_info["intensity_test"],
                                                  lum_labels[:54],
                                                  self.seq_info["intensity_test"],
                                                  lum_labels[54:],
                                                  self.seq_info["intensity_test"]])

                self.sess2vertical_transitions_dict[session_id] = np.diff(loc_labels_full) + 2 * loc_labels_full[1:]
                self.sess2brightness_transitions_dict[session_id] = np.diff(lum_labels_full)
                self.sess2seq[session_id] = self.ran_seq[:, seq_id]
                self.sess2seqid[session_id] = seq_id
                self.sess2seq_test[session_id] = np.concatenate([[1000, 2000, 3000, 4000, 5000],
                                                                 self.ran_seq[:54, seq_id],
                                                                 [1000, 2000, 3000, 4000, 5000],
                                                                 self.ran_seq[54:, seq_id],
                                                                 [1000, 2000, 3000, 4000, 5000]])
                temp = self.train_movie_snippets[:, self.ran_seq[:, seq_id]]
                self.sess2train_movie_dict[session_id] = temp
                self.sess2flattrain_movie_dict[session_id] = temp.reshape(2, 16200, 18, 16)
                concat = np.concatenate([self.test_movie_snippets,
                                         temp[:, :54, ...],
                                         self.test_movie_snippets,
                                         temp[:, 54:, ...],
                                         self.test_movie_snippets], axis=1)
                self.sess2movie_dict[session_id] = concat
                self.sess2flatmovie_dict[session_id] = concat.reshape(2, self.DUR_MOVIE, 18, 16)


######################################### ROC analyses #################################################################
def get_roc_curve(responses,
                  transitions,
                  target_transition,
                  offtarget_transition,
                  above_threshold,
                  num_bins=40):
    """
    Calculate ROC curve based on binned responses

    :param responses:
    :param transitions:
    :param target_transition:
    :param offtarget_transition:
    :param above_threshold:
    :param num_bins:
    :return:
    """
    ## get the indexes for target and non-target transitions
    positives_bool = transitions == target_transition
    if offtarget_transition == "all":
        negatives_bool = np.logical_not(positives_bool)
    else:
        negatives_bool = transitions == offtarget_transition
    ## sample equally spaced points on the response curve
    ds = np.linspace(responses.min(), responses.max(), num=num_bins)
    ## get a boolean array the size of the number of transitions/responses with True where the response is larger than threshold
    if above_threshold:
        predictions = np.asarray([responses > d for d in ds])
    else:
        predictions = np.asarray([responses <= d for d in ds])
    ## get fpr as the proportion of True predictions where label is negative
    fpr = np.dot(predictions.astype(int), negatives_bool.astype(int)) / negatives_bool.sum()
    tpr = np.dot(predictions.astype(int), positives_bool.astype(int)) / positives_bool.sum()
    ## get a function estimate of the ROC curve
    f = interpolate.interp1d(fpr[::-1], tpr[::-1])
    f_inverse = interpolate.interp1d(tpr[::-1], fpr[::-1])
    auc = f(np.arange(fpr.min(), fpr.max(), .001)).mean()
    return fpr, tpr, f, f_inverse, auc, ds


def get_ind_roc_curve(df, sess2vertical_transitions, binned_resp_dict, target_transition,
                      cell_type, offtarget_transition="all",
                      num_bins=40, above_threshold=True):
    """

    :param df:
    :param sess2vertical_transitions:
    :param binned_resp_dict:
    :param target_transition:
    :param cell_type:
    :param offtarget_transition:
    :param num_bins:
    :param above_threshold:
    :return:
    """
    tpr_by_nid = {}
    fpr_by_nid = {}
    auc_by_nid = {}
    functions_by_nid = {}
    inverse_functions_by_nid = {}
    thresholds_by_nid = {}
    curr_nids = df[df["group_assignment"] == cell_type]["neuron_id"].to_numpy()
    curr_sids = df[df["group_assignment"] == cell_type]["session_id"].to_numpy()
    for j, (curr_nid, curr_sid) in enumerate(zip(curr_nids, curr_sids)):
        fpr, tpr, f, f_inverse, auc, ds = get_roc_curve(binned_resp_dict[curr_nid],
                                                        sess2vertical_transitions[curr_sid],
                                                        target_transition, offtarget_transition,
                                                        above_threshold, num_bins)

        tpr_by_nid[curr_nid] = tpr
        fpr_by_nid[curr_nid] = fpr
        functions_by_nid[curr_nid] = f
        inverse_functions_by_nid[curr_nid] = f_inverse
        auc_by_nid[curr_nid] = auc
        thresholds_by_nid[curr_nid] = ds
    return tpr_by_nid, fpr_by_nid, auc_by_nid, functions_by_nid, inverse_functions_by_nid, thresholds_by_nid


def get_type_roc_curve(df, sess2vertical_transitions, binned_resp_dict, target_transition,
                       num_bins=40, above_threshold=True,):
    tpr_by_type = {}
    fpr_by_type = {}
    auc_by_type = {}
    functions_by_type = {}
    ds_by_type = {}
    for t in range(1, 33):

        fpr, tpr, f, auc, ds = get_roc_curve(binned_resp_dict[t],
                                             sess2vertical_transitions[t],
                                             target_transition, above_threshold, num_bins)

        tpr_by_type[t] = tpr
        fpr_by_type[t] = fpr
        auc_by_type[t] = auc
        functions_by_type[t] = f
        ds_by_type[t] = ds
    return tpr_by_type, fpr_by_type, auc_by_type, functions_by_type, ds_by_type


############################## Stats analyses ##########################################################################
import numpy as np
from copy import deepcopy


def cohens_d(sample_a, sample_b):
    sigma_a = np.std(sample_a)
    sigma_b = np.std(sample_b)
    n_a = len(sample_a)
    n_b = len(sample_b)
    standard_dev = np.sqrt(
        (((n_a-1) * sigma_a**2) + ((n_b-1) * sigma_b**2))/(n_a+n_b-2)
    )
    d = (sample_a.mean() - sample_b.mean())/standard_dev
    return abs(d)


def bootstrap_ci(sample_a, sample_b, n_rep=100, percentile=95):
    """
    Calculates the {percentile}th bootstrapped confidence interval for the delta
    of two data samples a and b.
    :param sample_a: np.array containing samples a
    :param sample_b: np.array containing samples b
    :param n_rep: number of repetitions for bootstrapping
    :param percentile: int in the interval [0, 100]
    :return:
    """
    n_a = len(sample_a)
    n_b = len(sample_b)
    lower_percentile = (100-percentile)/2
    upper_percentile = 100 - lower_percentile
    bootstrapped_diffs = np.asarray([
        np.random.choice(sample_a, size=n_a).mean()
        - np.random.choice(sample_b, size=n_b).mean() for _ in range(n_rep)
    ])
    lower_bound = np.percentile(bootstrapped_diffs, lower_percentile)
    upper_bound = np.percentile(bootstrapped_diffs, upper_percentile)
    return [lower_bound, upper_bound], bootstrapped_diffs


def perform_permutation_test(sample_a, sample_b, n_rep=100000, seed=8723):
    """
    Performs a permutation test on samples a and b
    :param sample_a:
    :param sample_b:
    :param n_rep:
    :param seed:
    :return:
    """
    np.random.seed(seed)
    size_a = len(sample_a)
    joined_samples = np.concatenate([sample_a, sample_b])
    orig_diff = abs(sample_a.mean() - sample_b.mean())
    permuted_diffs = np.zeros(n_rep)
    shuffling_array = deepcopy(joined_samples)
    for rep in range(n_rep):
        np.random.shuffle(shuffling_array)
        permuted_diffs[rep] = abs(shuffling_array[:size_a].mean() - shuffling_array[size_a:].mean())
    p_value = sum(permuted_diffs>orig_diff)/n_rep
    return permuted_diffs, orig_diff, p_value

