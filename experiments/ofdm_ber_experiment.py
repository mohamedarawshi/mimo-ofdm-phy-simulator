
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# Parameters
N = 64
CP_LENGTH = 16
NUM_OFDM_SYMBOLS = 5000

h = np.array([1, 0.9, 0.7, 0.5])

EbN0_dB_values = np.arange(0, 21, 2)

# Generate bits
bits = rng.integers(
    0, 2, size=(NUM_OFDM_SYMBOLS, 2*N)
)

# QPSK mapping
pairs = bits.reshape(NUM_OFDM_SYMBOLS, N, 2)

I = 1 - 2 * pairs[:, :, 1]
Q = 1 - 2 * pairs[:, :, 0]

X = (I + 1j*Q) / np.sqrt(2)

# IFFT
x = np.fft.ifft(X, axis=1)

# Add CP
x_cp = np.concatenate(
    (x[:, -CP_LENGTH:], x),
    axis=1
)

# Serialize OFDM symbols
tx_cp = x_cp.reshape(-1)
tx_no_cp = x.reshape(-1)

# Multipath channel
rx_cp_clean = np.convolve(tx_cp, h)[:len(tx_cp)]
rx_no_cp_clean = np.convolve(tx_no_cp, h)[:len(tx_no_cp)]

# Channel response
H = np.fft.fft(h, n=N)

ber_cp = []
ber_no_cp = []

for EbN0_dB in EbN0_dB_values:

    EbN0 = 10 ** (EbN0_dB / 10)

    # Noise variance for NumPy's IFFT normalization
    # (per useful time-domain sample)
    noise_variance = 1 / (2 * N * EbN0)

    # Noise for CP system
    noise_cp = np.sqrt(noise_variance / 2) * (
        rng.standard_normal(len(rx_cp_clean))
        + 1j*rng.standard_normal(len(rx_cp_clean))
    )

    # Noise for no-CP system
    noise_no_cp = np.sqrt(noise_variance / 2) * (
        rng.standard_normal(len(rx_no_cp_clean))
        + 1j*rng.standard_normal(len(rx_no_cp_clean))
    )

    rx_cp = rx_cp_clean + noise_cp
    rx_no_cp = rx_no_cp_clean + noise_no_cp

    # Receiver with CP
    rx_cp_blocks = rx_cp.reshape(
        NUM_OFDM_SYMBOLS, N + CP_LENGTH
    )

    y_cp = rx_cp_blocks[:, CP_LENGTH:]

    Y_cp = np.fft.fft(y_cp, axis=1)
    X_hat_cp = Y_cp / H

    # Receiver without CP
    y_no_cp = rx_no_cp.reshape(
        NUM_OFDM_SYMBOLS, N
    )

    Y_no_cp = np.fft.fft(y_no_cp, axis=1)
    X_hat_no_cp = Y_no_cp / H

    # Detect bits
    def detect(symbols):
        b0 = (symbols.imag < 0).astype(int)
        b1 = (symbols.real < 0).astype(int)
        return np.stack((b0, b1), axis=-1).reshape(
            NUM_OFDM_SYMBOLS, 2*N
        )

    bits_hat_cp = detect(X_hat_cp)
    bits_hat_no_cp = detect(X_hat_no_cp)

    # BER
    ber_cp.append(np.mean(bits != bits_hat_cp))
    ber_no_cp.append(np.mean(bits != bits_hat_no_cp))

    print(
        f"Eb/N0 = {EbN0_dB:2d} dB | "
        f"With CP = {ber_cp[-1]:.6f} | "
        f"Without CP = {ber_no_cp[-1]:.6f}"
    )

# Plot
plt.figure(figsize=(9, 6))

plt.semilogy(EbN0_dB_values, ber_cp, "o-", label="With CP")
plt.semilogy(
    EbN0_dB_values,
    ber_no_cp,
    "s-",
    label="Without CP"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")
plt.title("OFDM over Multipath: With vs Without CP")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("ofdm_cp_comparison.png", dpi=200)
plt.show()
