from torch import optim
from torch import nn
from copy import deepcopy
import torch
import numpy as np
import os


class MeiAcrossContrasts(nn.Module):
    """
    Create an MEI stimulus as a PyTorch Module for tracking gradients onto the
    scalars.
    """
    def __init__(self, contrast_values, mei_stim_private):
        """

        :param contrast_values: a torch Tensor with 2 scalar values that the
        MEI green and UV channels will be multiplied with
        :param mei_stim_private: MEI stimulus as torch Tensor
        """
        super().__init__()
        self.contrast_values = nn.Parameter(contrast_values, requires_grad=True)

        self.mei_stim_private = nn.Parameter(mei_stim_private, requires_grad=False)

    def forward(self):
        return torch.mul(
            torch.stack(
                [torch.ones_like(self.mei_stim_private[:, 0, ...]) * self.contrast_values[0],
                 torch.ones_like(self.mei_stim_private[:, 0, ...]) * self.contrast_values[1]
                 ], dim=1),
            self.mei_stim_private.squeeze()
        )


def trainer_fn(mei_contrast_gen, model_neuron, optimizer=optim.Adam, lr=1):
    """
    Trainer function for getting the gradient on the MEI with different contrasts
    """
    optimizer = optimizer(mei_contrast_gen.parameters(), lr=lr)
    loss = model_neuron(mei_contrast_gen())
    loss.backward()
    grad_val = deepcopy(mei_contrast_gen.contrast_values.grad)
    optimizer.zero_grad()
    return grad_val.detach().cpu().numpy().squeeze(), loss.detach().cpu().numpy().squeeze()

def get_gradient_grid(mei_stim, model_neuron, start=-1, stop=1, step_size=.1):
    """

    :param mei_stim:
    :param model_neuron:
    :param start:
    :param stop:
    :param step_size:
    :return:
    """
    n_channels = 2
    green_contrast_values = np.arange(start, stop+step_size, step_size)
    uv_contrast_values = np.arange(start, stop+step_size, step_size)
    grid = np.zeros((n_channels, len(green_contrast_values), len(uv_contrast_values)))
    resp_grid = np.zeros((len(green_contrast_values), len(uv_contrast_values)))
    norm_grid = np.zeros((len(green_contrast_values), len(uv_contrast_values)))

    for i, contrast_green in enumerate(np.arange(-1, 1+step_size, step_size)):
        for j, contrast_uv in enumerate(np.arange(-1, 1+step_size, step_size)):
            mei_contrast_gen = MeiAcrossContrasts(torch.Tensor([contrast_green, contrast_uv]), mei_stim)
            out, loss = trainer_fn(mei_contrast_gen, model_neuron, lr=.1)
            grid[0, i, j] = out[0]
            grid[1, i, j] = out[1]
            resp_grid[i, j] = loss
            norm_grid[i, j] = np.linalg.norm(grid[:, i, j])
    return grid, resp_grid, norm_grid, green_contrast_values, uv_contrast_values


def calculate_unit_vector(vector):
    magnitude = np.linalg.norm(vector)
    unit_vector = vector / magnitude
    return unit_vector


def calculate_angle(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_angle = dot_product / magnitude_product
    angle_in_radians = np.arccos(cosine_angle)
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees

def get_gradient_angle(gradient_grid):
    x_axis = [1, 0]
    angle_grid = np.zeros((gradient_grid.shape[1], gradient_grid.shape[2]))
    for i in range(gradient_grid.shape[1]):
        for j in range(gradient_grid.shape[2]):
            angle_grid[i, j] = calculate_angle(x_axis, gradient_grid[:, i, j])
    return angle_grid

