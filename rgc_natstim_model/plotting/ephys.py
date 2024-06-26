import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import seaborn as sns
from typing import List, Dict

from rgc_natstim_model.constants.plot_settings import cmap_colors
from rgc_natstim_model.constants.identifiers import example_nids

example_types = list(example_nids.keys())

def plot_imaging_ephys(resp_imaging: Dict,
                       resp_ephys: np.ndarray,
                       rgc_group: int,
                       sorting_idx: List,
                       opt_slice: slice,
                       ephys_idxs: List,
                       rc_dict_raster: Dict,
                       x_offset=np.float32
                       ):


    n_meis = 11
    with mpl.rc_context(rc_dict_raster):
        ########################## plot imaging responses #####################
        fig, ax1 = plt.subplots()
        temp = resp_imaging[rgc_group][..., opt_slice].mean(
            axis=-1
        )[:, sorting_idx]
        n_imaging = temp.shape[0]
        imaging_label = f'Imaging, n={n_imaging}'
        ax1.errorbar(np.arange(11)-x_offset,
                     temp.mean(axis=0),
                     yerr=temp.std(axis=0),
                     color='k',
                     fmt='o',
                     label=imaging_label
                     )
        ax1.set_ylabel('z-scored activity \n(Ca-imaging)',
                       color='k')
        ax1.tick_params(axis='y', colors='k')

        ########################## plot ephys responses #####################
        ax2 = ax1.twinx()
        n_ephys = len(ephys_idxs)
        ephys_label = f'Electrophysiology, n={n_ephys}'
        ax2.errorbar(np.arange(11) + x_offset,
                     resp_ephys[ephys_idxs, :].mean(axis=0),
                     yerr=resp_ephys[ephys_idxs, :].std(axis=0),
                     color='b',
                     fmt='o',
                     label=ephys_label
                     )

        ax2.set_ylabel('z-scored activity \n(patch-clamp)',
                       color='b')
        ax2.tick_params(axis='y', colors='b')

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.xaxis.set_major_locator(FixedLocator(range(n_meis)))
        ax1.set_xticklabels(example_types, rotation=90)
        ax1.set_xlabel("MEI stimuli")

        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)


def plot_raster(ax, rc_dict_raster, spike_time_dict):
    xlim = (-.1, 2.2)
    with mpl.rc_context(rc_dict_raster):
        lineoffsets = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
        lineheight = .5
        max_rep = max([len(v) for v in spike_time_dict.values()])
        max_rep = min([max_rep, 10])
        for i, (k, v) in enumerate(spike_time_dict.items()):
            t = int(k[4:])
            t_idx = example_types.index(t)
            len_ = len(v[:max_rep])
            ax.eventplot(v[:max_rep],
                         linelengths=.5,
                         lineoffsets=[max_rep * t_idx * lineheight + t_idx + i * lineheight
                                      for i in range(len_)],
                         color=cmap_colors[t - 1])
        # ax.fill_betweenx(plt.ylim(), 1, 1.66, color="k", alpha=.1, linewidth=0)
        ylim = ax.get_ylim()
        ax.fill_betweenx(ylim, xlim[0], 1, color="k", alpha=.1, linewidth=0)
        ax.fill_betweenx(ylim, 1.66, xlim[1], color="k", alpha=.1, linewidth=0)
        sns.despine()
        plt.xlim(xlim)
        plt.xlabel("Time [s]")


def plot_rate(
        ax: plt.axis,
        rc_dict_raster: Dict,
        spike_rates,
        opt_slice: slice,
):
    xlim = (-.1, 2.2)
    t_stop = 3
    t_locs = np.linspace(0, t_stop, t_stop * 30)

    zorder_dict = {1: 0, 5: 1, 10: 2, 18: 3, 20: 4, 21: 5, 23: 6, 24: 7, 28: 10, 31: 8, 32: 9}
    dummy_spikes = [0, 0.25, 0.4, 0.5]
    with mpl.rc_context(rc_dict_raster):

        for t in example_types:
            m = np.mean(np.stack([fr for fr in spike_rates[t]], axis=0), axis=0)
            std = np.std(np.stack([fr for fr in spike_rates[t]], axis=0), axis=0)

            plt.plot(t_locs[opt_slice], m[opt_slice],
                     color=cmap_colors[t - 1], zorder=zorder_dict[t],
                     )
            plt.plot(t_locs, m,
                     color=cmap_colors[t - 1], zorder=zorder_dict[t],
                     linewidth=.5)

            plt.xlabel("Time (s)")
            plt.ylabel("Firing rate")
            plt.xlim(xlim)
        ylim = ax.get_ylim()
        ax.fill_betweenx(ylim, xlim[0], 1, color="k", alpha=.1, linewidth=0)
        ax.fill_betweenx(ylim, 1.66, xlim[1], color="k", alpha=.1, linewidth=0)
        sns.despine()
