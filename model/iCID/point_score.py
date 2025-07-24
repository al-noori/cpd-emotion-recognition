import numpy as np
from numpy.lib._stride_tricks_impl import sliding_window_view

from model.iCID.aNNEspace import aNNEspace


def point_score(Y, psi, window, seed):
    """
    Calculate each point dissimilarity score

    Parameters:
    - Y: np array of shape (n, d), the data points
    - psi: int, the number of partitions within each partitioning
    - window: int, the size of the moving window
    """
    Y = (Y - np.min(Y)) / (1.0 * (np.max(Y) - np.min(Y)))  # Normalize Y to [0, 1]
    Y[np.isnan(Y)] = 0.5  # Replace NaNs with 0
    type = 'NormalisedKernel'

    Sdata = Y
    data = Y

    t = 200

    ndata = aNNEspace(Sdata, data, psi, t, seed)

    # index each segmentation
    n = int(Y.shape[0])
      # for CPD:---------------

    mdata = []
    startL = np.arange(0, n - 2 * window)
    endL = np.arange(window - 1, n - window - 1)
    startR = np.arange(window + 1, n - window + 1)
    endR = np.arange(2 * window - 1, n)
    index = list(zip(startL, endL, startR, endR))
    index = np.array(index).flatten()
    for i in range(0, len(index) - 1, 2):
        start, end = index[i] , index[i + 1]
        segment = ndata[start:end, :]
        mdata.append(np.mean(segment, axis=0))

    mdata = np.array(mdata)

    # k-nn comparison with k = 1
    k = 1
    score = []

    norms = np.linalg.norm(mdata, axis=1)
    dot_products = np.sum(mdata[1::2] * mdata[0::2], axis=1)
    sim = dot_products / (norms[1::2] * norms[0::2])
    score = 1 - sim
    score = np.insert(score, 0, 0)
    pscore = np.zeros(int(n))
    pscore[window: window + len(score)] = score


    # Normalize scores to [0, 1]
    pscore = np.array(pscore, dtype=np.float32)
    denom = np.max(pscore) - np.min(pscore)
    if denom != 0:
        pscore = (pscore - np.min(pscore)) / denom
    else:
        pscore.fill(0)
    return pscore