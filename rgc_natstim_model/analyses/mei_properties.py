import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def calculate_angle(dataframe: pd.DataFrame,
                    target_column: str,
                    diag_column: str,
                    offdiag_column: str,
                    inplace: bool =True):
    """
    Calculate the angle about the diagonal (i.e. UV=green) for a given MEI property (e.g. size), using the projection onto the diagonal and the off-diagonal.
    
    Parameters:
        dataframe (pd.DataFrame): The input dataframe.
        target_column (str): The name of the new column to be inserted (e.g. angle_size).
        diag_column (str): The name of the column holding the projection onto the diagonal.
        offdiag_column (str): The name of the column holding the projection onto the off-diagonal.
        inplace (bool, optional): If True, the new column is inserted in-place. If False, the new column is returned.
    
    Returns:
        pd.Series or None: The new column if inplace=False, otherwise None.
    """
    column = abs(np.arctan(dataframe[offdiag_column] / dataframe[diag_column]))
    if inplace:
        dataframe.insert(dataframe.shape[1], target_column, column)
    else:
        return column

def get_temporal_contrast(neuron_id,
                          chans,
                          df: pd.DataFrame,
                          sd_factor=1,
                          distance=5,
                          temp_key="temporal_kernel",
                          spat_key="spatial_kernel",
                          sing_key="sigular_value",
                          recon_key="recon_mei",
                          use_center_mask=True):
    """
    Fetch the SVD components of a neuron's MEI and calculate temporal contrast as the signed
    difference between the last and the second-to-last peak. To get absolute amplitude, multiply with singular value
    Args:
        neuron_id (int): The ID of the neuron.
        chans (list): List of channels.
        df (pd.DataFrame): The DataFrame containing the MEI data.
        sd_factor (float, optional): The standard deviation factor. Defaults to 1.
        distance (int, optional): The distance between peaks, passed to find_peaks function. Defaults to 5.
        temp_key (str, optional): The key for the temporal kernel in the DataFrame. Defaults to "temporal_kernel".
        spat_key (str, optional): The key for the spatial kernel in the DataFrame. Defaults to "spatial_kernel".
        sing_key (str, optional): The key for the singular value in the DataFrame. Defaults to "singular_value".
        recon_key (str, optional): The key for the rank-1 MEI reconstructed from spatial and temporal component in the DataFrame. Defaults to "recon_mei".
        use_center_mask (bool, optional): Whether to use the center mask. Defaults to True.
    Returns:
        tuple: A tuple containing three dictionaries:
            - contrast_dict (dict): A dictionary containing the temporal contrast for each channel.
            - contrast_dict_abs (dict): A dictionary containing the absolute temporal contrast for each channel.
            - peaks_dict (dict): A dictionary containing the locations of the negative and positive peaks for each channel.
    """

    temp, spat, s, recon_mei = df[[temp_key, spat_key, sing_key, recon_key]].loc[neuron_id].to_numpy()
    contrast_dict = {"green": None, "uv": None}
    contrast_dict_abs = {"green": None, "uv": None}
    peaks_dict = {"green": None, "uv": None}
    _, n_rows, n_cols = spat.shape
    center_row = n_rows // 2 - 1
    center_col = n_cols // 2 - 1
    ## find peak location for UV
    c = 1
    threshold = np.mean(temp[c]) + sd_factor * np.std(temp[c])

    # find positive peak
    pos_threshold = np.min([threshold, temp[c].max() - .05])
    peak_locs, peak_props = find_peaks(temp[c], height=pos_threshold, distance=distance)
    pos_peak_loc = peak_locs[-1]
    pos_peak_ampl = peak_props["peak_heights"][-1]
    # find negative peak
    neg_threshold = np.min([threshold, (-temp[c]).max() - .05])

    peak_locs, peak_props = find_peaks(-temp[c], height=neg_threshold, distance=distance)
    neg_peak_loc = peak_locs[-1]
    sorted_peaks = np.sort([neg_peak_loc, pos_peak_loc])
    for c, k in zip(chans, contrast_dict.keys()):
        if use_center_mask:
            if len(df["mask_center_{}".format(k)].loc[neuron_id]) > 0:
                weights = np.tile(df["mask_center_{}".format(k)].loc[neuron_id],
                                  (50, 1, 1))

                mean_mei = np.average(recon_mei[c], weights=weights, axis=(1, 2))

                # print(mean_mei[sorted_peaks[1]])
                contrast_dict_abs[k] = mean_mei[sorted_peaks[1]] - mean_mei[sorted_peaks[0]]
            else:
                contrast_dict_abs[k] = np.nan
        else:
            contrast_dict_abs[k] = recon_mei[c][sorted_peaks[1]][center_row, center_col] - \
                                   recon_mei[c][sorted_peaks[0]][center_row, center_col]
        contrast_dict[k] = temp[c][sorted_peaks[1]] - temp[c][sorted_peaks[0]]
        peaks_dict[k] = [neg_peak_loc, pos_peak_loc]
    return contrast_dict, contrast_dict_abs, peaks_dict

