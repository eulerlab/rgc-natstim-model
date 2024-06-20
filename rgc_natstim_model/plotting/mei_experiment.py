from collections import OrderedDict
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from . import cmap_colors
from scipy.interpolate import interp1d
import seaborn as sns

def plot_resp_per_position(dataframe, type_to_nid, neuron_id, sorting_idx,
                           opt_slice, time_line_rec, alphas, example_types,
                           fig_kwargs = {}, gridspec_kw={},
                           col_key="spat_ord_resp",
                           weight_key ="spatial_weighting_function",
                           rc_context_dict={}, types_to_plot=[],
                          fig = None, axes=None, linestyle="-", linewidth=1):
    ordered_type_to_nid = OrderedDict(zip(example_types[sorting_idx],
                                          [type_to_nid[example_types[i]] for i in sorting_idx]))
    with mpl.rc_context(rc_context_dict):
        if fig is None:
            fig, axes = plt.subplots(**fig_kwargs, gridspec_kw=gridspec_kw)
        normed_spat_weight = deepcopy(dataframe[weight_key].loc[neuron_id])
        normed_spat_weight /= normed_spat_weight.max()
        resp = dataframe[col_key].loc[neuron_id]
        resp_sorted = resp[sorting_idx]
        for k, (type_, nid) in enumerate(ordered_type_to_nid.items()):
            if type_ in types_to_plot or len(types_to_plot)==0:
                for pos in range(25):
                    (i, j) = np.unravel_index(pos, shape=(5, 5), order="C")
                    axes[i, j].plot(time_line_rec, resp_sorted[k, i, j, :],
                                    color="k",
                                    alpha = .1,
                                    linestyle=linestyle,
                                    linewidth=linewidth
                                   )
                    axes[i, j].plot(time_line_rec[opt_slice],
                                    resp_sorted[k, i, j, opt_slice], cmap_colors[type_-1],
                                    alpha = alphas[k],
                                    label=type_,
                                   linestyle=linestyle,
                                   linewidth=linewidth)

                    axes[i, j].axis("off")
        plt.tight_layout()
        return fig


import matplotlib.ticker as ticker
@ticker.FuncFormatter
def major_formatter(x, pos):
    return "%.1f" % x


def plot_responses(time_line_mod, opt_slice, resp, type_to_nid,
                   rc_context_dict={}, alphas=None, aggregate_fun=np.mean,
                   is_model=False, fig_kwargs={}, sorting_idx=[], ylim=None,
                   xlabels=True, indicate_stim=False):
    subscript = "ret" if not (is_model) else "mod"
    if resp.shape[0] > 1: # more than 1 cell
        ylabel_0 = "$\langle r_{%s}\\rangle_{s, i}$" % subscript
        # if aggregate_fun == np.max:
        #     ylabel_1 = "$max(\langle r_{%s}\\rangle_{s, i})_t$" % subscript
        # elif aggregate_fun == np.mean:
        #     ylabel_1 = "$\langle r_{%s}\\rangle_{s, i, t}$" % subscript
    else:
        ylabel_0 = "$\langle r_{%s}\\rangle_{s}$" % subscript
        # if aggregate_fun == np.max:
        #     ylabel_1 = "$max(\langle r_{%s}\\rangle_{s})_t$" % subscript
        # elif aggregate_fun == np.mean:
        #     ylabel_1 = "$\langle r_{%s}\\rangle_{s, t}$" % subscript
    if len(sorting_idx) == 0:
        sorting_idx = [i for i in range(len(type_to_nid))]
    with mpl.rc_context(rc_context_dict):
        if alphas is None:
            alphas = [1 for k in range(len(type_to_nid))]
        fig, axes = plt.subplots(**fig_kwargs)
        mean_across_time = resp[..., opt_slice].mean(-1)
        #normalized_mean_across_time = mean_across_time/mean_across_time.max(axis=1)
        for i, (k, (type_, nid)) in enumerate(zip(sorting_idx, type_to_nid.items())):
            #print(k, type_)
            # if not (len(time_line) == len(time_line_mod)):
            #     interp_funcs = [interp1d(time_line, resp[i, k, :]) for i in range(resp.shape[0])]
            #     temp = np.asarray([interp_f(time_line_mod) for interp_f in interp_funcs])
            #
            # else:
            #     temp = resp[:, k, :]
            temp = resp[:, k, :]
            mean_across_neurons = temp.mean(axis=0)

            axes[0].plot(time_line_mod, mean_across_neurons, color="k", alpha=.1)
            axes[0].plot(time_line_mod[opt_slice], mean_across_neurons[opt_slice],
                         cmap_colors[type_ - 1], label=type_, alpha=alphas[k])
            axes[1].scatter(i, aggregate_fun(mean_across_neurons[opt_slice]), color=cmap_colors[type_ - 1],
                            alpha=alphas[k])
            axes[1].errorbar(x=i, y=aggregate_fun(mean_across_neurons[opt_slice]),
                             yerr=aggregate_fun(temp[:, opt_slice], axis=1).std(),
                             color=cmap_colors[type_ - 1], ls="none", alpha=alphas[k])
        axes[0].yaxis.set_major_locator(MaxNLocator(4))
        if xlabels:
            axes[1].set_xticks(range(11))
            axes[1].set_xticklabels(type_to_nid.keys(), rotation=90, fontsize=5)
            axes[1].set_xlabel("Type MEIs")
            axes[0].set_xlabel("Time [s]")
            axes[0].set_xticks([0, 1, 2])
        else:
            axes[1].set_xticks(range(11))
            axes[1].set_xticklabels(["" for i in range(11)])
            axes[0].set_xticks([0, 1, 2])
            axes[0].set_xticklabels(["" for i in range(3)])
        #axes[0].set_ylabel(ylabel_0)
        axes[0].yaxis.set_major_formatter(major_formatter)
        if ylim is not None:
            for ax in fig.get_axes():
                ax.set_ylim(ylim)
        plt.sca(axes[0])
        if indicate_stim:
            ys = plt.ylim()
            plt.hlines(y=ys[1], xmin=0, xmax=1.66, color="k")
            axes[0].fill_betweenx(ys, x1=1, x2=1.66, color="k",
                                  alpha=.2, linewidth=0)
        sns.despine()
        plt.tight_layout()
    return fig