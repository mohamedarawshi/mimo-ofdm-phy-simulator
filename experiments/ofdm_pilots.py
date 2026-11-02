
import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------
# 1. OFDM parameters
# ---------------------------------
N = 64
CP_LENGTH = 16
PILOT_SPACING = 8

# Pilot and data subcarrier indices
pilot_indices = np.arange(0, N, PILOT_SPACING)
data_indices = np.setdiff1d(
    np.arange(N), pilot_indices
)

print("Pilot indices:", pilot_indices)
print("Number of pilots:", len(pilot_indices))
print("Number of data subcarriers:", len(data_indices))

# ---------------------------------
# 2. Generate information bits
# ---------------------------------
num_bits = 2 * len(data_indices)

bits = rng.integers(0, 2, size=num_bits)

# ---------------------------------
# 3. QPSK modulation
# ---------------------------------
pairs = bits.reshape(-1, 2)

I = 1 - 2 * pairs[:, 1]
Q = 1 - 2 * pairs[:, 0]

data_symbols = (I + 1j * Q) / np.sqrt(2)

# ---------------------------------
# 4. Insert data and pilots
# ---------------------------------
X = np.zeros(N, dtype=complex)

# Data symbols
X[data_indices] = data_symbols

# Known pilot symbols
X[pilot_indices] = 1 + 0j

print("Information bits:", num_bits)
print("First 16 frequency-domain symbols:")
print(X[:16])

# ---------------------------------
# 5. IFFT and cyclic prefix
# ---------------------------------
x = np.fft.ifft(X)

cp = x[-CP_LENGTH:]
x_with_cp = np.concatenate((cp, x))

print("OFDM useful samples:", len(x))
print("Transmitted samples with CP:", len(x_with_cp))

# ---------------------------------
# 6. Multipath channel + AWGN
# ---------------------------------

h = np.array([1, 0.9, 0.7, 0.5])

EbN0_dB = 15
EbN0 = 10 ** (EbN0_dB / 10)

# All frequency-domain symbols have unit energy.
# There are 56 QPSK data symbols = 112 data bits.
# Noise is normalized to the useful data-bit energy,
# excluding CP and pilot overhead.
N0 = 1 / (2 * EbN0)
noise_variance = N0 / N

# Apply multipath
rx_clean = np.convolve(x_with_cp, h)

# Generate complex AWGN in the time domain
noise = np.sqrt(noise_variance / 2) * (
    rng.standard_normal(len(rx_clean))
    + 1j * rng.standard_normal(len(rx_clean))
)

received = rx_clean + noise

# ---------------------------------
# 7. Remove CP and apply FFT
# ---------------------------------

y = received[CP_LENGTH:CP_LENGTH + N]

Y = np.fft.fft(y)

print("Received frequency-domain symbols:", len(Y))


# ---------------------------------
# 8. LS channel estimation
# ---------------------------------

H_pilot_est = (
    Y[pilot_indices] / X[pilot_indices]
)

# True channel response (for evaluation only)
H_true = np.fft.fft(h, n=N)

print("\nPilot channel estimates:")

for i, k in enumerate(pilot_indices):
    print(
        f"k={k:2d} | "
        f"True H={H_true[k]:.3f} | "
        f"Estimated H={H_pilot_est[i]:.3f}"
    )

# Estimation error on pilot subcarriers
pilot_mse = np.mean(
    np.abs(
        H_pilot_est - H_true[pilot_indices]
    ) ** 2
)

print("\nPilot channel estimation MSE:", pilot_mse)



# ---------------------------------
# 9. Channel interpolation
# ---------------------------------

# Extend pilots periodically for the FFT boundary
extended_indices = np.append(pilot_indices, N)

extended_estimates = np.append(
    H_pilot_est,
    H_pilot_est[0]
)

# Interpolate real and imaginary components
H_est_real = np.interp(
    np.arange(N),
    extended_indices,
    extended_estimates.real
)

H_est_imag = np.interp(
    np.arange(N),
    extended_indices,
    extended_estimates.imag
)

H_est = H_est_real + 1j * H_est_imag

# Calculate MSE across all 64 subcarriers
channel_mse = np.mean(np.abs(H_est - H_true)**2)

print("Full channel estimation MSE:", channel_mse)


# ---------------------------------
# 10. Plot channel estimation
# ---------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(
    np.arange(N),
    np.abs(H_true),
    label="True channel",
    linewidth=2
)

plt.plot(
    np.arange(N),
    np.abs(H_est),
    "--",
    label="Estimated channel"
)

plt.scatter(
    pilot_indices,
    np.abs(H_pilot_est),
    label="Pilot estimates",
    zorder=3
)

plt.xlabel("Subcarrier index")
plt.ylabel("Channel magnitude |H[k]|")
plt.title("OFDM Pilot-Based Channel Estimation")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("channel_estimation.png", dpi=200)
plt.show()


# ---------------------------------
# 11. Equalization using estimated H
# ---------------------------------

# Equalize data subcarriers only
X_hat_LS = Y[data_indices] / H_est[data_indices]

# Ideal reference: perfect channel knowledge
X_hat_perfect = Y[data_indices] / H_true[data_indices]

# ---------------------------------
# 12. QPSK hard-decision detector
# ---------------------------------

def detect_qpsk(symbols):
    detected_bits = np.empty(2 * len(symbols), dtype=int)

    detected_bits[0::2] = (symbols.imag < 0).astype(int)
    detected_bits[1::2] = (symbols.real < 0).astype(int)

    return detected_bits

bits_LS = detect_qpsk(X_hat_LS)
bits_perfect = detect_qpsk(X_hat_perfect)

# ---------------------------------
# 13. Calculate bit errors and BER
# ---------------------------------

errors_LS = np.sum(bits_LS != bits)
errors_perfect = np.sum(bits_perfect != bits)

BER_LS = errors_LS / len(bits)
BER_perfect = errors_perfect / len(bits)

print("\n--- Receiver Performance ---")
print(f"LS channel estimation: {errors_LS} errors, BER={BER_LS:.6f}")
print(f"Perfect CSI:           {errors_perfect} errors, BER={BER_perfect:.6f}")
