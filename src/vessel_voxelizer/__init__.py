from typing import Union

import numpy as np

from ._vessel_vox import __voxelize
from .utils import load_vessels


def voxelize(volume_shape: np.ndarray, volume_origin: np.ndarray, volume_spacing: Union[np.ndarray, float],
             vessel_positions: np.ndarray, vessel_radii: np.ndarray) -> np.ndarray:
    """Voxelization of vessel segments.

    Parameters
    ----------
    volume_shape : np.ndarray
        The shape of the volume that the vessels should be written into.
    volume_origin : np..ndarray
        Origin coordinates of the volume (corner of first voxel).
    volume_spacing : float, np.ndarray
        Voxel side length.
    vessel_positions : np.ndarray
        Start and end points of the vessels. Shape: (N, 2, 3)
    vessel_radii : np.ndarray
        Radii of the vessel segments. Shape: (N,)
    """

    volume = np.zeros(volume_shape, dtype=np.float32)
    volume_origin = volume_origin.astype(np.float32)
    if isinstance(volume_spacing, float):
        volume_spacing = np.array([volume_spacing, volume_spacing, volume_spacing], dtype=np.float32)
    else:
        volume_spacing = volume_spacing.astype(np.float32)
    vessel_positions = vessel_positions.astype(np.float32)
    vessel_radii = vessel_radii.astype(np.float32)

    vessel_bounds = np.sort(vessel_positions, axis=1)
    vessel_bounds = vessel_bounds + np.hstack(
        (-vessel_radii.reshape(-1, 1, 1),
         vessel_radii.reshape(-1, 1, 1))
    )

    __voxelize(volume, volume_origin, volume_spacing, vessel_positions, vessel_bounds, vessel_radii)

    return volume
