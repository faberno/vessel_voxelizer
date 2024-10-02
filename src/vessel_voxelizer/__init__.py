import cupy as cp

from ._vessel_vox import voxelize as _voxelize
from .utils import load_vessels


def voxelize(volume: cp.ndarray, volume_start: cp.ndarray, volume_spacing: float,
             vessel_positions: cp.ndarray, vessel_radii: cp.ndarray) -> cp.ndarray:
    """
    """
    # compute the bounding boxes of the vessels
    vessel_bounds = cp.sort(vessel_positions, axis=1) + cp.hstack(
        (-vessel_radii.reshape(-1, 1, 1),
         vessel_radii.reshape(-1, 1, 1))
    )

    return _voxelize(volume, volume_start, volume_spacing, vessel_positions, vessel_bounds, vessel_radii)
