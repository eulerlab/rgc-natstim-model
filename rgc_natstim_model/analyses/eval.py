from rgc_natstim_model.constants.imaging import scan_frequency
import numpy as np
from scipy.stats import pearsonr as lincorr


def calculate_nat_reliability_index(df_row):
    traces, traces_times, triggertimes = df_row[['natural_detrended_traces',
                                             'natural_traces_times',
                                             'natural_trigger_times']]
    test_triggers = [triggertimes[0], triggertimes[59], triggertimes[118]] #this assumes a specific presentation order! valid for mouse cam
    idxs = [np.nonzero(np.isclose(traces_times, np.ones_like(traces_times) * s, atol=1e-01))[0][0]
           for s in test_triggers]
    test_traces = np.asarray(
        [traces[i:i + int(np.ceil(scan_frequency * 25))] for i in idxs]
    )
    test_traces = np.transpose(test_traces)
    numerator = np.var(np.mean(test_traces, axis=-1), axis=0)
    denom = np.mean(np.var(test_traces, axis=0), axis=-1)
    movie_qi = numerator / denom
    return movie_qi


def calculate_correlations(responses, predictions, resp_idxs, model_readout_idxs):
    """
    Takes as input the test responses (3 reps) and the model predictions,
    and calculates the avg. correlation, and the correlation to avg.
    Correlation measure is linear correlation (Pearson)

    Inputs:
    ------
    responses:   torch.Tensor    shape n_neurons x n_reps x n_timesteps
    predictions: np.Array        shape n_timesteps-30 x n_neurons
    """
    n_neurons = responses.shape[0]
    corrs = np.zeros((n_neurons, 3))
    corrs_to_avg = np.zeros(n_neurons)
    responses = np.transpose(responses, (2, 0, 1)) # to shape n_timesteps x n_neurons x n_reps
    for neuron in range(n_neurons):
        for rep in range(3):
            corrs[neuron, rep] = lincorr(responses[30:, resp_idxs[neuron], rep], predictions[:, model_readout_idxs[neuron]]).statistic
        corrs_to_avg[neuron] = lincorr(responses[30:, resp_idxs[neuron]].mean(axis=-1), predictions[:, model_readout_idxs[neuron]]).statistic
    return corrs, corrs_to_avg