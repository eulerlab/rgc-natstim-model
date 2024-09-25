import numpy as np

def get_weighted_resp(df, neuron_id, col_key="resp_per_loc"):
    """
    Get the spatially weighted response of a neuron to the MEI stimuli.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the responses and spatial weighting functions.
    neuron_id : int
        Index of the neuron in the DataFrame.
    col_key : str
        Key for the column containing the responses (recorded or model responses).

    Returns
    -------
    resp : np.ndarray
        Weighted response of the neuron.
    """

    weights = df["spatial_weighting_function"].loc[neuron_id]
    normed_weights = weights / weights.sum()

    resp = np.einsum('ij,ij...',
                     normed_weights,
                     np.moveaxis(df[col_key].loc[neuron_id], 0, 3)).transpose()[np.newaxis]
    return resp