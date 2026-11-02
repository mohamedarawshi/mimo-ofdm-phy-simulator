import numpy as np

def mmse_equalize(y_data, h_est_k, noise_variance, symbol_energy=1.0):
    """Y[rx,ofdm,k], H[k,rx,tx] -> Xhat[tx,ofdm,k]."""
    if noise_variance < 0 or symbol_energy <= 0:
        raise ValueError("invalid noise variance or symbol energy")
    y = np.transpose(y_data, (2, 0, 1))
    hh = np.conjugate(np.swapaxes(h_est_k, 1, 2))
    a = hh @ h_est_k + (noise_variance / symbol_energy) * np.eye(h_est_k.shape[2])
    x = np.linalg.solve(a, hh @ y)
    return np.transpose(x, (1, 2, 0))

def zf_equalize(y_data, h_est_k):
    """Per-subcarrier ZF via Moore-Penrose pseudoinverse."""
    y = np.transpose(y_data, (2, 0, 1))
    x = np.linalg.pinv(h_est_k) @ y
    return np.transpose(x, (1, 2, 0))
