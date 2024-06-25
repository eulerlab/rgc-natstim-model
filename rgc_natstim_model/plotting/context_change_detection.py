from scipy.stats import gaussian_kde
import numpy as np
def kde_for_display(data, bins, n_samples, norm=True, bw_method=None):
    counts, bin_edges = np.histogram(data, bins)
    bin_width = np.diff(bin_edges)[0]
    x_locs = bin_edges[:-1]+.5*bin_width
    samples = np.random.choice(x_locs, size=n_samples, p=counts/counts.sum())
    dens_estimator = gaussian_kde(samples, bw_method=bw_method)
    density = dens_estimator(x_locs)
    if norm:
        density/=density.sum()
    return x_locs, density