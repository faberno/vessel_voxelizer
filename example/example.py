import numpy as np
import vessel_voxelizer as vv
import matplotlib.pyplot as plt
import os

def main():

    # choose gpu array module
    module = 'cupy'  # 'torch'

    # read the vessel data
    vessel_positions, vessel_radii = vv.load_vessels(os.path.abspath("../files/example_vessels.csv"))

    # define the voxel grid
    volume_spacing = 0.01
    volume_bounds = np.array([[-0.1, 4.4], [-0.1, 3.9], [-0.1, 1.54]])
    volume_shape = np.round((volume_bounds[:, 1] - volume_bounds[:, 0]) / volume_spacing).astype(int)

    if module == 'cupy':
        import cupy as cp

        # move data to gpu
        volume_start = cp.asarray(volume_bounds[:, 0], dtype=cp.float32)
        volume = cp.zeros(volume_shape, dtype=np.float32)
        vessel_positions = cp.asarray(vessel_positions)
        vessel_radii = cp.asarray(vessel_radii)

        # run voxelization
        vv.voxelize(volume, volume_start, volume_spacing, vessel_positions, vessel_radii)

        # move results to cpu
        volume = volume.get()

    elif module == 'torch':
        import torch

        # move data to gpu
        volume_start = torch.as_tensor(volume_bounds[:, 0], dtype=torch.float32, device='cuda')
        volume = torch.zeros(tuple(volume_shape), dtype=torch.float32, device='cuda')
        vessel_positions = torch.as_tensor(vessel_positions, device='cuda')
        vessel_radii = torch.as_tensor(vessel_radii, device='cuda')

        # run voxelization
        vv.voxelize(volume, volume_start, volume_spacing, vessel_positions, vessel_radii)

        # move results to cpu
        volume = volume.cpu().numpy()

    else:
        raise NotImplementedError(f"Module {module} is not supported yet.")


    # plot results
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    axs[0].imshow(np.rot90(volume.max(0), 1))
    axs[1].imshow(np.rot90(volume.max(1), 1))
    axs[2].imshow(volume.max(2))
    axs[3].axis('off')
    plt.show()

if __name__ == "__main__":
    main()
