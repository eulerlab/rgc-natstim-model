import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def rgb_from_stim(stim, color_channel_axis=0, target_cc_axis=-1, red=0):
    rgb = np.moveaxis(stim, color_channel_axis, target_cc_axis)
    red_channel = np.zeros((*rgb.shape[:-1], 1), dtype=int)
    if red:
        # if red == 1, set the red channel to the mean of blue and green everywhere
        red_channel = rgb.mean(axis=-1)[..., np.newaxis]
    rgb = np.concatenate([red_channel, rgb], axis=-1)
    return rgb

def transform(bip_dog, norm_channels_separately, vmin=None, vmax=None):
    """
    Maps a stimulus to the range [0, 1]
    :param bip_dog: shape channels x time x height x width
    :param: norm_channels_separately: boolean whether to normalize channels individually or jointly
    :param: vmin/vmax: optional parameters to set the limits of the color map
    :return:
    """
    if norm_channels_separately:
        for c in range(bip_dog.shape[0]):
            vmin = bip_dog[c].min()
            vmax = bip_dog[c].max()
            abs_max = max([abs(vmin), abs(vmax)])
            bip_dog[c] = bip_dog[c] / abs_max
            bip_dog[c] = bip_dog[c] * .5
            bip_dog[c] = bip_dog[c] + .5
            bip_dog[c] -= bip_dog[c].min()
            bip_dog[c] = bip_dog[c] / bip_dog[c].max()

    else:
        if vmin is None:
            vmin = bip_dog.min()
        if vmax is None:
            vmax = bip_dog.max()
        abs_max = max([abs(vmin), abs(vmax)])
        bip_dog = bip_dog/abs_max  # min val >= -1, max val <=1
        bip_dog = bip_dog * .5
        bip_dog = bip_dog + .5

    return bip_dog


def bip_dog_to_rgb(bip_dog, norm_channels_separately=False,
                   transform_kwargs=dict(), red=0):
    """

    :param bip_dog: kernel as np array, shape: channels x time x height x width
    :param norm_channels_separately: boolean whether to normalize channels individually or jointly
    :param transform_kwargs:
    :param red: boolean, whether to set the red channel to 0 or to the mean of B&G
    :return:
    """
    bip_dog = transform(bip_dog, norm_channels_separately,
                        **transform_kwargs)
    rgb = rgb_from_stim(bip_dog, red=red)
    return rgb

def space_time_plot(stim, x_loc=7, cmap_green="Greys_r",
                            cmap_uv="Greys_r",
                            axes=None, fig=None,
                            normalize="jointly",
                            t_start=0,
                    red=1):
    """
    creates a single column, three row plot of the space-time evolution of MEIs
    shown in greyscale for green and UV, and shown in RGB for the overlay of the
    two colors (with R = (B+G)/2 everywhere), at a vertical slice x_loc
    :param stim:
    :param x_loc:
    :param cmap_green:
    :param cmap_uv:
    :param axes:
    :param fig:
    :param normalize:
    :param t_start:
    :param red:
    :return:
    """
    stim_as_rgb = bip_dog_to_rgb(stim, norm_channels_separately=False, red=red)
    abs_max = max([abs(stim.min()), abs(stim.max())])
    norm = Normalize(vmin=-abs_max, vmax=abs_max)
    if normalize == "jointly":
        norm_green = norm
        norm_uv = norm
    else:
        abs_max = max([abs(stim[0].min()), abs(stim[0].max())])
        norm_green = Normalize(vmin=-abs_max, vmax=abs_max)
        abs_max = max([abs(stim[1].min()), abs(stim[1].max())])
        norm_uv = Normalize(vmin=-abs_max, vmax=abs_max)
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(3, 3))
    axes[0].imshow(stim[0, t_start:, :, x_loc].transpose(), norm=norm_green, cmap=cmap_green)
    axes[1].imshow(stim[1, t_start:, :, x_loc].transpose(), norm=norm_uv, cmap=cmap_uv)
    axes[2].imshow(np.swapaxes(stim_as_rgb[t_start:, :, x_loc, :], 0, 1))
    for ax in fig.get_axes():
        ax.axis("off")
    return fig




'''Dataframe level plotting'''
def space_time_plot_from_df(df, neuron_id, **space_time_kwargs):
    """

    fetches the MEI from a dataframe indexed by neuron_id, and calls
    space_time_plot with the given params
    :param df:
    :param neuron_id:
    :param space_time_kwargs:
    :return:
    """
    stim = df["mei"].loc[neuron_id]
    fig = space_time_plot(stim, **space_time_kwargs)
    return fig

def spatial_temporal_chromatic_plot_from_df(df, neuron_ids,
                                            green_cmap="binary_r",
                                            uv_cmap="binary_r", t_start=0,
                                            x_loc=7, y_loc=7, slice_vertical=True):
    n_cols_fig = len(neuron_ids)
    fig, axes = plt.subplots(4, n_cols_fig, figsize=(n_cols_fig, 4), sharey="row")
    if len(axes.shape) == 1:
        axes = axes[:, np.newaxis]
    for idx, neuron_id in enumerate(neuron_ids):
        spatial_kernel_green = df["spatial_kernel_green"].loc[neuron_id]
        spatial_kernel_uv = df["spatial_kernel_uv"].loc[neuron_id]
        temporal_kernels = df["temporal_kernel"].loc[neuron_id]
        abs_max = max([abs(spatial_kernel_green.max()),
                       abs(spatial_kernel_green.min()),
                       abs(spatial_kernel_uv.max()),
                       abs(spatial_kernel_uv.min())])
        norm = Normalize(vmin=-abs_max, vmax=abs_max)
        for chan, chan_key, cmap in zip([0, 1], ["green", "uv"], [green_cmap, uv_cmap]):
            axes[chan, idx].imshow(df["spatial_kernel_{}".format(chan_key)].loc[neuron_id],
                                         cmap=cmap, norm=norm)

            try:
                center_contour = df['_'.join(["center_contour", chan_key])].loc[neuron_id]
                axes[chan, idx].plot(center_contour[0][:, 1],
                                     center_contour[0][:, 0], color="k", alpha=.5, linewidth=1)
            except IndexError:
                print("No center contour found for neuron_id {}: passing".format(neuron_id))
                pass
            try:
                surround_contour = df['_'.join(["surround_contour", chan_key])].loc[neuron_id]
                axes[chan, idx].plot(surround_contour[0][:, 1],
                                     surround_contour[0][:, 0], color="k", linestyle="dotted", alpha=.5,
                                     linewidth=1)
            except IndexError:
                print("No suround contour found for neuron_id {}: passing".format(neuron_id))
                pass

        axes[2, idx].plot(temporal_kernels[0] * df["singular_values"].loc[neuron_id][0][0],
                          color="darkgreen")
        axes[2, idx].plot(temporal_kernels[1] * df["singular_values"].loc[neuron_id][1][0],
                          color="darkviolet")
        stim = df["mei"].loc[neuron_id]
        stim_as_rgb = bip_dog_to_rgb(stim, norm_channels_separately=False, red=1)
        if slice_vertical:
            axes[3, idx].imshow(np.swapaxes(stim_as_rgb[t_start:, :, x_loc, :], 0, 1))
        else:
            axes[3, idx].imshow(np.swapaxes(stim_as_rgb[t_start:, y_loc, :, :], 0, 1))
        # axes[0, idx].set_title(neuron_id)
    for ax in fig.get_axes():
        ax.axis("off")
    return fig
