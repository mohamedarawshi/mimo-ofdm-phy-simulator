
import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------
# 1. Generate QPSK symbols
# ---------------------------------

NUM_SYMBOLS = 100000

bits = rng.integers(0, 2, 2 * NUM_SYMBOLS)
pairs = bits.reshape(-1, 2)

I = 1 - 2 * pairs[:, 1]
Q = 1 - 2 * pairs[:, 0]

X = (I + 1j * Q) / np.sqrt(2)

# ---------------------------------
# 2. Channel and noise
# ---------------------------------

H = 0.1 + 0.05j

EbN0_dB = 10
EbN0 = 10 ** (EbN0_dB / 10)

Es = 1
Eb = Es / 2
N0 = Eb / EbN0

W = np.sqrt(N0 / 2) * (
    rng.standard_normal(NUM_SYMBOLS)
    + 1j * rng.standard_normal(NUM_SYMBOLS)
)

Y = H * X + W

# ---------------------------------
# 3. ZF equalization
# ---------------------------------

G_zf = 1 / H
X_zf = G_zf * Y

# ---------------------------------
# 4. MMSE equalization
# ---------------------------------

G_mmse = np.conj(H) / (np.abs(H)**2 + N0)

X_mmse = G_mmse * Y

# ---------------------------------
# 5. Calculate symbol MSE
# ---------------------------------

mse_zf = np.mean(np.abs(X_zf - X)**2)
mse_mmse = np.mean(np.abs(X_mmse - X)**2)

print("ZF symbol MSE:", mse_zf)
print("MMSE symbol MSE:", mse_mmse)

# ---------------------------------
# 6. QPSK hard-decision detector
# ---------------------------------

def detect_qpsk(symbols):
    detected = np.empty(2 * len(symbols), dtype=int)
    detected[0::2] = (symbols.imag < 0).astype(int)
    detected[1::2] = (symbols.real < 0).astype(int)
    return detected

bits_zf = detect_qpsk(X_zf)
bits_mmse = detect_qpsk(X_mmse)

ber_zf = np.mean(bits_zf != bits)
ber_mmse = np.mean(bits_mmse != bits)

print("ZF BER:", ber_zf)
print("MMSE BER:", ber_mmse)
