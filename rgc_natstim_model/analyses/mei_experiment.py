import numpy as np

def get_weighted_resp(df, neuron_id, col_key="resp_per_loc"):
    weights = df["spatial_weighting_function"].loc[neuron_id]
    normed_weights = weights / weights.sum()

    resp = np.einsum('ij,ij...',
                     normed_weights,
                     np.moveaxis(df[col_key].loc[neuron_id], 0, 3)).transpose()[np.newaxis]
    return resp