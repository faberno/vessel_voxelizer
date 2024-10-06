import numpy as np
import pandas as pd


def load_vessels(path):
    """Read vessel data from csv file.

    The csv should be comma separated with headers

    node_id | x | y | z | radius | parent_id

    and every row describes a vessel segment node.
    A root should have parent_id=-1.

    Parameters
    ----------
    path : str
        Path of the csv file.

    Parameters
    ----------
    np.ndarray
        Position of vessel segments. Shape (N, 2, 3)
    np.ndarray
        Radii of the vessel segments. Shape: (N,)
    """
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

    return vessel_positions, vessel_radii