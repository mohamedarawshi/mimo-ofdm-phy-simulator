import numpy as np

def qpsk_modulate(bits):
    """Gray QPSK; first bit controls Q, second controls I; Es=1."""
    bits = np.asarray(bits)
    if bits.size % 2 or np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must contain an even number of binary values")
    pairs = bits.reshape(-1, 2)
    return ((1 - 2 * pairs[:, 1].astype(float)) + 1j * (1 - 2 * pairs[:, 0].astype(float))) / np.sqrt(2)

def qpsk_detect(symbols):
    """Hard QPSK decisions, flattened in bit-pair order."""
    flat = np.asarray(symbols).reshape(-1)
    bits = np.empty(flat.size * 2, dtype=np.uint8)
    bits[0::2] = flat.imag < 0
    bits[1::2] = flat.real < 0
    return bits
