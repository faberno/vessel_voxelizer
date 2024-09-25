import numpy as np
import pandas as pd


def load_vessels(path):
    vessels = pd.read_csv(path)

    # number of vessel segments
    N = len(vessels) - (vessels['parent_id'] == -1).sum()

    ids = vessels['node_id'].to_numpy()
    positions = vessels[['x', 'y', 'z']].to_numpy()
    radii = vessels['radius'].to_numpy()
    parent_id = vessels['parent_id'].to_numpy()

    vessel_positions = np.zeros((N, 2, 3))
    vessel_radii = np.zeros(N)

    i_vessel = 0
    for i in range(len(vessels)):
        parent = parent_id[i]
        if parent == -1:
            continue
        else:
            vessel_positions[i_vessel] = positions[[parent_id[i], ids[i]]]
            vessel_radii[i_vessel] = radii[i]
            i_vessel += 1

    # compute the bounding boxes of the vessels
    vessel_bounds = np.sort(vessel_positions, axis=1) + np.hstack(
        (-vessel_radii.reshape(-1, 1, 1),
         vessel_radii.reshape(-1, 1, 1))
    )

    return vessel_positions, vessel_radii, vessel_bounds