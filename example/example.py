import numpy as np
import cupy as cp
import vessel_voxelizer as vv
import matplotlib.pyplot as plt
import os
def main():

    # read the vessel data
    vessel_positions, vessel_radii, vessel_bounds = vv.load_vessels(os.path.abspath("../files/example_vessels.csv"))

    # define the voxel grid
    volume_spacing = 0.01
    volume_bounds = np.array([[-0.1, 4.4], [-0.1, 3.9], [-0.1, 1.54]])

    volume_shape = np.round((volume_bounds[:, 1] - volume_bounds[:, 0]) / volume_spacing).astype(int)
    volume_start = cp.asarray(volume_bounds[:, 0], dtype=cp.float32)
    volume = cp.zeros(volume_shape, dtype=np.float32)
    vessel_positions, vessel_radii, vessel_bounds = cp.asarray(vessel_positions), cp.asarray(vessel_radii), cp.asarray(vessel_bounds)

    vv.voxelize(volume, volume_start, volume_spacing, vessel_positions, vessel_bounds, vessel_radii)
    volume = volume.get()

    fig, axs = plt.subplots(3, 1, figsize=(5, 12))
    axs[0].imshow(np.rot90(volume.max(0), 1))
    axs[1].imshow(np.rot90(volume.max(1), 1))
    axs[2].imshow(volume.max(2))
    plt.show()


if __name__ == "__main__":
    main()