import numpy as np
import os
import torch
import SimpleITK as sitk
import torch.nn.functional as F

def fast_bilinear_interpolation(input_array, x_indices, y_indices):

    # Rescale the x and y indices to match the input array shape
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)

    x = x_indices - x0
    y = y_indices - y0

    output = (
        input_array[x0, y0] * (1 - x) * (1 - y)
        + input_array[x1, y0] * x * (1 - y)
        + input_array[x0, y1] * (1 - x) * y
        + input_array[x1, y1] * x * y
    )
    return output

def bilinear_interpolation(input_array, x_indices, y_indices, mode='nearest'):

    assert mode in ['nearest', 'bilinear']

    # Rescale the x and y indices to match the input array shape
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5

    # Reshape the x and y indices to match the input array shape and move to CPU
    x_indices = x_indices.reshape(input_array.shape).detach().cpu()
    y_indices = y_indices.reshape(input_array.shape).detach().cpu()

    coord_grid = torch.stack([y_indices, x_indices], axis=2).unsqueeze(0).cuda()
    im_shape = input_array.shape

    max_extent = (
        torch.tensor(
            im_shape[::-1], dtype=coord_grid.dtype, device=coord_grid.device
        )
        - 1
    )

    coord_grid = 2 * (coord_grid / max_extent) - 1

    output = F.grid_sample(
        input_array.unsqueeze(0).unsqueeze(0),
        coord_grid,
        mode=mode,
        padding_mode='border',
        align_corners=True,
    )
    return output.squeeze()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_coordinate_tensor(dims=(28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(2)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=2)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 2])

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor


def make_masked_coordinate_tensor(mask, dims=(28, 28)):
    """Make a coordinate tensor."""

    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(2)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor)
    coordinate_tensor = torch.stack(coordinate_tensor, dim=2)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 2])
    coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor
