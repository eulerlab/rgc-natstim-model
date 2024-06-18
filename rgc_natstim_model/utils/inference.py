import numpy as np
import torch
from functools import partial


def _get_responses(model, stimulus, max_stim_per_batch):
    n_stim = stimulus.shape[0]
    if n_stim > max_stim_per_batch:
        all_responses = []
        for batch in image_batches(max_stim_per_batch, stimulus, n_stim):
            responses = reduce_fn(model(batch)).detach().cpu().numpy()
            all_responses.append(responses)
        all_responses = np.vstack(all_responses)
    else:
        all_responses = model(stimulus).detach().cpu().numpy()
    return all_responses


def image_batches(batch_size, stimulus, num_stims):
    for batch_start in np.arange(0, num_stims, batch_size):
        batch_end = np.minimum(batch_start + batch_size, num_stims)
        images = [stimulus[i] for i in range(batch_start, batch_end)]
        yield torch.stack(images)


def get_model_responses(model, stimulus, session_id,
                        batch_size=1000):
    """ For a given model, stimulus and restriction (dataset & ensemble), return a
    dictionary of neuron_ids to responses to the stimulus"""

    responses = _get_responses(partial(model, data_key=session_id), stimulus,
                               batch_size)
    return responses
