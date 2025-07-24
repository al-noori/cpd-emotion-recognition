from model.iCID.best_psi import best_psi
from model.iCID.best_threshold import best_threshold
from scipy.signal import find_peaks

def run_iCID(Y, win_size=40, alpha=1, seed = 42):
    pscore, _, _ = best_psi(Y, win_size, seed = seed)
    threshold, _ = best_threshold(pscore, alpha)
    indices = find_peaks(pscore, height=threshold)[0]
    return indices, pscore, threshold
