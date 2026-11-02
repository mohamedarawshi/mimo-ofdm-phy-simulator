import numpy as np

def frequency_response(taps, n_subcarriers):
    """Return H[k,rx,tx] from taps[rx,tx,delay]."""
    return np.moveaxis(np.fft.fft(taps, n=n_subcarriers, axis=-1), -1, 0)

def apply_mimo_fir(tx_streams, taps):
    """Continuous, causal, linear FIR convolution; truncate to input length."""
    tx_streams = np.asarray(tx_streams)
    taps = np.asarray(taps)
    nrx, ntx, _ = taps.shape
    if tx_streams.shape[0] != ntx:
        raise ValueError("transmit antenna count mismatch")
    out = np.zeros((nrx, tx_streams.shape[1]), dtype=complex)
    for rx in range(nrx):
        for tx in range(ntx):
            out[rx] += np.convolve(tx_streams[tx], taps[rx, tx])[:tx_streams.shape[1]]
    return out
