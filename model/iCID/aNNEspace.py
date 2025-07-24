import numpy as np
from sklearn.neighbors import NearestNeighbors


# nearest Neighbor Embedding (NNE) space transformation function
def aNNEspace(Sdata, data, psi, t, seed):
    """
    Randomized approximate nearest neighbor embedding.

    Parameters:
    - Sdata: numpy array of shape (sn, d)
    - data: numpy array of shape (n, d)
    - psi: int, number of partitions within each partitioning
    - t: int, finite number of partitionings

    Returns:
    - ndata: numpy array of shape (n, t * psi), the transformed feature space
    """
    sn = Sdata.shape[0]
    n = data.shape[0]
    ndata = np.empty((n, t * psi), dtype=np.float32)

    for i in range(t):
        # randomly sample psi indices from Sdata without replacement
        rng = np.random.default_rng(seed + i)
        sub_indices = rng.choice(sn, size=psi, replace=False)
        tdata = Sdata[sub_indices, :]

        # compute pairwise distances between sampled tdata and data
        neigh = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(tdata)
        _, center_idx = neigh.kneighbors(data)

        # build one-hot
        z = np.zeros((n, psi), dtype=np.float16)
        z[np.arange(n), center_idx.flatten()] = 1
        ndata[:, i*psi:(i+1)*psi] = z

    return ndata
