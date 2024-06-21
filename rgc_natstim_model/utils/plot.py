import matplotlib.pyplot as plt


def save_this(path, fname, png=True, svg=True, pdf = True, fig=None, transparent=True):
    if fig is None:
        fig = plt.gcf()
    if png:
        plt.savefig(path+fname+'.png', dpi=300, transparent=transparent,
                   bbox_inches="tight")
    if svg:
        plt.savefig(path+fname+'.svg', dpi=300, transparent=transparent,
                   bbox_inches="tight")
    if pdf:
        print("pdf")
        plt.savefig(path + fname + '.pdf', dpi=300, transparent=transparent,
                    bbox_inches="tight")