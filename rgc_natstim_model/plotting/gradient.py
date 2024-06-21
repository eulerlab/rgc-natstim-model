import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from rgc_natstim_model.analyses.gradient import calculate_unit_vector
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, FormatStrFormatter, FixedLocator


class ColorMapper:
    def __init__(self, cmap_name, vmin=0, vmax=1):
        self.cmap_name = cmap_name
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = plt.get_cmap(self.cmap_name)
        self.norm = plt.Normalize(vmin=self.vmin, vmax=self.vmax)

    def get_color(self, number):
        color = self.cmap(self.norm(number))
        return color


def plot_vector_field(green_contrast_values, uv_contrast_values,
                      gradient_norm_grid, gradient_grid, rc_dict={}):
    cm = ColorMapper("cool", vmin=gradient_norm_grid.min(),
                     vmax=gradient_norm_grid.max())
    with mpl.rc_context(rc_dict):
        fig = plt.figure(figsize=(6, 6))
        for i, contrast_green in enumerate(green_contrast_values):
            for j, contrast_uv in enumerate(uv_contrast_values):
                unit_vec = calculate_unit_vector(gradient_grid[:, i, j]) * .1
                plt.arrow(contrast_green, contrast_uv, unit_vec[0], unit_vec[1],
                          fill=False,
                          head_width=.03, color=cm.get_color(gradient_norm_grid[i, j]))
        plt.xlabel("green magnitude")
        plt.ylabel("uv magnitude")
        plt.colorbar(plt.cm.ScalarMappable(norm=cm.norm, cmap=cm.cmap),
                            )
        sns.despine()
    return fig


def plot_vector_field_resp_iso(x, y,
                      gradient_norm_dict, gradient_dict, resp_dict,
                               neuron_id, normalize_response=False,
                               colour_norm = False, rc_dict={},
                               cmap="hsv"):
    def format_xticks(value, pos):
        return int(value)
    Z = resp_dict[neuron_id].transpose()
    if normalize_response:
        Z=Z/Z.max() * 100
    gradient_norm_grid = gradient_norm_dict[neuron_id]
    gradient_grid = gradient_dict[neuron_id][:, 1:-1, 1:-1]
    X, Y = np.meshgrid(x, x)

    # Define levels for isoresponse lines
    levels = np.linspace(Z.min(), Z.max(), 25)
    print(levels.min())
    print(levels.max())
    #levels = np.linspace(0, 100, 26)
    print(gradient_norm_grid.min())
    print(gradient_norm_grid.max())
    cm = ColorMapper("cool", vmin=gradient_norm_grid.min(),
                     vmax=gradient_norm_grid.max())

    with mpl.rc_context(rc_dict):
        fig = plt.figure()

        # Create a contour plot with isoresponse lines

        # cont_lines = plt.contour(X, Y, Z, levels=levels, cmap=cmap, zorder=200)  # Change cmap to the desired colormap
        # plt.gca().clabel(cont_lines, inline=True, fmt='%1.0f',
        #                  levels = levels[::2], colors="k", fontsize=5, zorder=400)

        plt.contourf(X, Y, Z, levels=levels, cmap=cmap, zorder=200)  # Change cmap to the desired colormap
        cont_lines = plt.contour(X,Y,Z, levels=levels, cmap='jet_r',zorder=300)
        plt.gca().clabel(cont_lines, inline=True, fmt='%1.0f',
                         levels = cont_lines.levels[::2], colors="k", fontsize=5, zorder=400)
        ax = plt.gca()
        ax.set_aspect("equal")


        if colour_norm:
            NotImplementedError()
            # for i, contrast_green in enumerate(x):
            #     for j, contrast_uv in enumerate(y):
            #         unit_vec = calculate_unit_vector(gradient_grid[:, i, j]) * .1
            #         plt.arrow(contrast_green, contrast_uv, unit_vec[0], unit_vec[1],
            #                   fill=False,
            #                   head_width=.03, color=cm.get_color(gradient_norm_grid[i, j]))
            # plt.colorbar(plt.cm.ScalarMappable(norm=cm.norm, cmap=cm.cmap), FuncFormatter()
                                 # )
        else:
            for i, contrast_green in enumerate(x[1:-1]):
                for j, contrast_uv in enumerate(y[1:-1]):
                    unit_vec = calculate_unit_vector(gradient_grid[:, i, j]) * .1
                    ax.arrow(contrast_green, contrast_uv, unit_vec[0], unit_vec[1],
                              fill=True, linewidth=.5,
                              head_width=.03, color="grey", zorder=300)
        ax.set_xlabel("Green contrast")
        ax.set_ylabel("UV contrast")
        ax.xaxis.set_major_locator(FixedLocator([-1, 0, 1]))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.yaxis.set_major_locator(FixedLocator([-1, 0, 1]))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        sns.despine()
    return fig


def plot_resp_diff(nonlin_dict, lin_dict, neuron_id,  green_contrast_values,
                   rc_dict={}):
    def format_xticks(value, pos):
        return f'{xtick_labels[pos]:.1f}'

    xtick_labels = green_contrast_values

    with mpl.rc_context(rc_dict):
        a = nonlin_dict[neuron_id].transpose()
        a = 100*a/a.max()
        b = lin_dict[neuron_id].transpose()
        b = 100*b / b.max()
        diff = a-b
        abs_max = np.max([abs(diff.min()), abs(diff.max())])
        norm=Normalize(vmin=-abs_max, vmax=abs_max)
        fig, ax = plt.subplots()
        im = ax.imshow(diff,
                         origin="lower", cmap="RdBu_r", norm=norm)

        # Set custom tick positions and labels
        ax.set_xticks(range(len(green_contrast_values)))
        ax.set_yticks(range(len(green_contrast_values)))
        ax.set_xticklabels(xtick_labels, rotation=90)
        ax.set_yticklabels(xtick_labels)
        ax.xaxis.set_major_formatter(FuncFormatter(format_xticks))
        ax.yaxis.set_major_formatter(FuncFormatter(format_xticks))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)  # Adjust the size and padding as needed
        # Add colorbar to the new axis
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('diff. in response (% of max.)')

        # Set axis labels
        ax.set_xlabel('Green contrast')
        ax.set_ylabel('UV contrast')

    sns.despine()
    return fig


