import importlib
from typing import Union

from ._vessel_vox import __voxelize
from .utils import load_vessels

ndarray = Union["cp.ndarray", "torch.Tensor"]


def voxelize(volume: ndarray, volume_origin: ndarray, volume_spacing: Union[ndarray, float],
             vessel_positions: ndarray, vessel_radii: ndarray) -> ndarray:
    """Voxelization of vessel segments.

    Parameters
    ----------
    volume : cp.ndarray, torch.Tensor
        The volume that the vessels should be written into.
    volume_origin : cp.ndarray, torch.Tensor
        Origin coordinates of the volume (corner of first voxel).
    volume_spacing : float, cp.ndarray, torch.Tensor
        Voxel side length.
    vessel_positions : cp.ndarray, torch.Tensor
        Start and end points of the vessels. Shape: (N, 2, 3)
    vessel_radii : cp.ndarray, torch.Tensor
        Radii of the vessel segments. Shape: (N,)
    """

    module = volume.__class__.__module__  # currently either cupy or torch
    xp = importlib.import_module(module)

    # compute the bounding boxes of the vessels
    if module == "cupy":
        vessel_bounds = xp.sort(vessel_positions, axis=1)

        if isinstance(volume_spacing, float):
            volume_spacing = xp.array([volume_spacing, volume_spacing, volume_spacing])
    elif module == "torch":
        vessel_bounds = xp.sort(vessel_positions, axis=1)[0]  # torch.sort returns a tuple

        if isinstance(volume_spacing, float):
            volume_spacing = xp.tensor([volume_spacing, volume_spacing, volume_spacing], device='cuda')
    else:
        raise NotImplementedError(f"Module {module} is currently not supported."
                                  f"Please open a issue: https://github.com/faberno/vessel_voxelizer/issues")
    vessel_bounds = vessel_bounds + xp.hstack(
        (-vessel_radii.reshape(-1, 1, 1),
         vessel_radii.reshape(-1, 1, 1))
    )

    __voxelize(volume, volume_origin, volume_spacing, vessel_positions, vessel_bounds, vessel_radii)
