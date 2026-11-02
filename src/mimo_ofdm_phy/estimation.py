import numpy as np

def orthogonal_pilots():
    """P[tx,pilot_symbol] with P P^H = 2 I."""
    return np.array([[1, 1], [1, -1]], dtype=complex)

def ls_channel_estimate(y_pilot, pilot_matrix):
    """LS H[rx,tx,k] from Y[rx,pilot_symbol,k]. Requires full row rank P."""
    y = np.asarray(y_pilot)
    p = np.asarray(pilot_matrix)
    if y.ndim != 3 or p.ndim != 2 or y.shape[1] != p.shape[1]:
        raise ValueError("pilot observation and matrix dimensions mismatch")
    if np.linalg.matrix_rank(p) != p.shape[0]:
        raise ValueError("pilot matrix must have full row rank")
    return np.einsum('rpk,pt->rtk', y, np.linalg.pinv(p))

def dft_denoise(h_frequency, channel_length):
    """Project H[rx,tx,k] onto first channel_length delay taps."""
    n = h_frequency.shape[-1]
    if not 1 <= channel_length <= n:
        raise ValueError("channel_length must be in [1, n_subcarriers]")
    taps = np.fft.ifft(h_frequency, axis=-1)
    taps[..., channel_length:] = 0
    return np.fft.fft(taps, axis=-1)
