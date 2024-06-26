import numpy as np
from elephant.statistics import optimal_kernel_bandwidth, instantaneous_rate
from neo.core import SpikeTrain
from quantities import s, ms
from elephant import kernels
import quantities as qu
from rgc_natstim_model.constants.identifiers import example_nids

example_types = list(example_nids.keys())


def get_spike_raster(raster, cell_idx):
    spike_time_dict = dict()
    for i in range(11):
        key = raster["outdata"][0][cell_idx][1].squeeze()[i][0][0]
        temp = raster["outdata"][0][cell_idx][1].squeeze()[i][1].squeeze()
        n_rep = len(temp)
        spikes_by_rep = []
        for rep in range(n_rep):
            dummy = temp[rep][0].squeeze()*10**-4
            if len(dummy.shape)==0:
                dummy = dummy[np.newaxis]
            spikes_by_rep.append(dummy)
        spike_time_dict[key] = spikes_by_rep
    return spike_time_dict


def get_rates(spike_time_dict,
              kernel='auto',
              sampling_period = 33*ms,
              opt_slice=slice(30, 50),
              t_start=0,
              t_stop = 3,
              max_rep=18):

    trains_by_mei = dict()
    rates_by_mei = dict()
    z_scored_rates = dict()
    for k, v in spike_time_dict.items():
        # k is type name, v is list with spike times per repetition
        t = int(k[4:])
        n_rep = max_rep#len(v[:10])
        trains_by_mei[t] = []
        rates_by_mei[t] = np.zeros((len(v), 90))
        z_scored_rates[t] = np.zeros((len(v), 90))
        for c, e in enumerate(v): # iterate over spike times per repetition
            train = SpikeTrain(e[e>t_start]*s, t_stop=t_stop)
            if kernel == 'auto':
                try:
                    rate = instantaneous_rate(train,
                                              kernel=kernel,
                                              sampling_period=sampling_period,
                                              t_start = 0*s,
                                              t_stop=t_stop*s)
                    print(rate.annotations['kernel']['sigma'])
                except ValueError:
                    print('Could not estimate optimal kernel width')
                    kernel = kernels.GaussianKernel(sigma=75 * qu.ms)
                    rate = instantaneous_rate(train,
                                              kernel=kernel,
                                              sampling_period=sampling_period,
                                              t_start=0 * s,
                                              t_stop=t_stop * s)
            else:
                rate = instantaneous_rate(train,
                                          kernel=kernel,
                                          sampling_period=sampling_period,
                                          t_start=0 * s,
                                          t_stop=t_stop * s)

            trains_by_mei[t].append(train)
            rates_by_mei[t][c] = (np.asarray([el[0] for el in rate.magnitude]))
    flat_rate = np.concatenate([rates_by_mei[t].reshape(-1) for t in example_types])
    mean_rate = flat_rate.mean()
    std_rate = flat_rate.std()
    [z_scored_rates.update({t: (rates_by_mei[t]-mean_rate)/std_rate}) for t in example_types];
    return trains_by_mei, rates_by_mei, z_scored_rates