import numpy as np
from joblib import Parallel, delayed
from model.iCID.point_score import point_score

# Entropy Approximation according to Pincus, 1991 

def _max_dist(x_i, x_j):
    return np.max(np.abs(x_i - x_j))

def _phi(m, r, data):
    N = len(data)
    x = np.array([data[i:i+m] for i in range(N - m + 1)])
    C = []
    for i in range(0, x.shape[0]):
        count = 0
        for j in range(0,len(x)):
            if _max_dist(x[i], x[j]) <= r:
                count += 1
        C.append(count / (N - m + 1))

    C = np.array(C)
    return np.sum(np.log(C)) / (N - m + 1)

def approximate_entropy(data, m, r):
    data = np.ravel(data)  # flatten in case input shape is (N,1)
    return _phi(m, r, data) - _phi(m + 1, r, data)

def best_psi(Y, window, seed):
    psi_list = np.array([2, 4, 8, 16, 32, 64])

    results = Parallel(n_jobs=-1)(delayed(_score_and_entropy)(Y, psi, window, seed) for psi in psi_list)
    pscore_list, ent_list = zip(*results)
    best_idx = np.argmin(ent_list)
    return pscore_list[best_idx], ent_list[best_idx], psi_list[best_idx]


def _score_and_entropy(Y, psi, window, seed):
    pscore = point_score(Y, psi, window, seed)
  #  ent = approximate_entropy(pscore, m=2, r=0.2 * np.std(pscore))

    ent = np.var(pscore)  # Using variance as a proxy for entropy
    return np.array(pscore, dtype=np.float32), ent