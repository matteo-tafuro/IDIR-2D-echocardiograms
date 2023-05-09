import torch
import numpy as np
from utils import general

def run_inference(img2warp, trained_model, return_dvf=False):
    """
    Apply a trained Implicit Neural Representation model to warp an input image and optionally return the Displacement
    Vector Field (DVF) used for the warping.

    Args:
        img2warp: The input image to warp of shape [H, W, C].
        trained_model: The trained Implicit Neural Representation model to use for warping.
        return_dvf: Whether to return the Displacement Vector Field (DVF) used for warping. Defaults to False.

    Returns:
        warped_img: The warped input image of shape [H, W, C].
        dvf: The Displacement Vector Field (DVF) used for warping of shape [2, H, W] if `return_dvf=True`.

    """

    # Create the coordinate tensor with shape [H*W, 2] from the image dimensions
    coordinate_tensor = general.make_coordinate_tensor(dims=img2warp.shape)

    # Use the trained model to warp the input image based on the coordinate tensor
    warped_img = trained_model(coordinate_tensor, moving_image=img2warp.cuda(), output_shape=img2warp.shape).astype(np.uint8)

    # Optionally return the Displacement Vector Field (DVF) used for the warping
    if return_dvf:
        # Use the trained model to transform the coordinate tensor
        coordinate_tensor_transformed = trained_model.network(coordinate_tensor)

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(coordinate_tensor_transformed, coordinate_tensor)
        x_indices, y_indices = coord_temp[:, 0],  coord_temp[:, 1],

        # Create an empty input array of the same shape as the input image
        input_array = np.zeros_like(img2warp)

        # Rescale the x and y indices to match the input array shape
        x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
        y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5

        # Reshape the x and y indices to match the input array shape and move to CPU
        x_indices = x_indices.reshape(input_array.shape).detach().cpu()
        y_indices = y_indices.reshape(input_array.shape).detach().cpu()

        # Stack the x and y indices to create the Displacement Vector Field (DVF)
        displaced_grid = np.stack([y_indices, y_indices], axis=0)

        # Create the identity grid with the same shape as the input array
        identity_grid = np.indices(input_array.shape).astype(np.float32)

        # Compute the Displacement Vector Field (DVF) as the difference between the displaced and identity grids
        dvf = displaced_grid - identity_grid

        return warped_img, dvf

    # Return the warped image if `return_dvf=False`
    return warped_img
