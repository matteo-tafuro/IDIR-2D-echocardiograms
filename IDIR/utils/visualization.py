import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import SimpleITK as sitk

def plot_grid(dvf, step = 10, ax=None, **kwargs):
    """
    Plot a deformed grid defined by a displacement field.
    Args:
        dvf: Displacement field of shape [2,H,W]
        step: Spacing between grid lines (default=10)
        ax: Matplotlib axis (default=None)
        **kwargs: Additional keyword arguments to be passed to matplotlib.collections.LineCollection
    """
    # as seen in https://stackoverflow.com/questions/47295473/how-to-plot-using-matplotlib-python-colahs-deformed-grid
    u,v = dvf[0], dvf[1]
    grid_x, grid_y = np.meshgrid(np.arange(0, u.shape[1], 1), np.arange(0, v.shape[0], 1))
    f = lambda x, y : (x + u, y + v)
    distx, disty = f(grid_x, grid_y)
    ax = ax or plt.gca()
    segs1 = np.stack((distx[::step, ::step], disty[::step, ::step]), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.invert_yaxis()
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()


def plot_vectorfield(dvf, step = 10, ax=None, **kwargs):
    """
    Plot a vector field.
    Args:
        dvf: Displacement field of shape [2,H,W]
        step: Spacing between arrows (default=10)
        ax: Matplotlib axis (default=None)
        **kwargs: Additional keyword arguments to be passed to matplotlib.pyplot.quiver
    """
    x,y = np.meshgrid(np.arange(0, dvf.shape[2]), np.arange(0,dvf.shape[1]))
    u,v = dvf[0], dvf[1]
    ax = ax or plt.gca()
    ax.quiver(x[::step, ::step],y[::step, ::step],u[::step, ::step],v[::step, ::step], **kwargs)
    ax.invert_yaxis()

def plot_jacobian_determinant(jacobian, jacobian_masked, ax=None, **kwargs):
    """
    Plot the jacobian determinant and the masked jacobian determinant
    
    Args:
        jacobian: Jacobian determinant of shape [H,W]
        jacobian_masked: Masked jacobian determinant of shape [H,W] where the values are less than zero
        ax: Matplotlib axis object to plot on
        **kwargs: Additional keyword arguments to be passed to matplotlib.pyplot.imshow
    """
    ax = ax or plt.gca()
    # Plot jacobian determinant
    im1 = ax.imshow(jacobian, cmap='PuOr', norm=mpl.colors.Normalize(vmin=0, vmax=2))
    # Plot critical points, i.e. where the jacobian determinant is less than zero
    im2 = ax.imshow(jacobian_masked, cmap='bwr', norm=mpl.colors.Normalize(vmin=-1e8, vmax=-1e7))
    # hide ticks
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # Plot colormap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax, ticks=[0.0, 0.5, 1.0, 1.5, 2.0], orientation='horizontal')
    # Add title
    ax.set_title('Jacobian determinant')

def compute_jacobian_determinant(dvf_np):
    """
    Compute the jacobian determinant of a displacement field
    Args:
        dvf_np: Displacement field of shape [2,H,W]
    Returns:
        jacobian: Jacobian determinant of shape [H,W]
        jacobian_masked: Masked jacobian determinant of shape [H,W] where the values are less than zero
    """
    #Convert the numpy array to a HxW image with each pixel being a 2D vector and compute the jacobian determinant volume
    sitk_displacement_field = sitk.GetImageFromArray(dvf_np.transpose(1,2,0), isVector=True)
    jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
    jacobian = sitk.GetArrayViewFromImage(jacobian_det_volume,).astype(np.float16)
    # Find the critical points, i.e. where the jacobian determinant is less than zero
    jacobian_masked = np.ma.masked_where((jacobian > 0), jacobian)

    return jacobian, jacobian_masked