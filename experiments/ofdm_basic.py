
import numpy as np

rng = np.random.default_rng(42)

# ---------------------------------
# 1. Parameters
# ---------------------------------
N = 64
CP_LENGTH = 16

# ---------------------------------
# 2. Generate random bits
# ---------------------------------
bits = rng.integers(0, 2, size=2*N)

# ---------------------------------
# 3. QPSK modulation
# ---------------------------------
pairs = bits.reshape(-1, 2)

I = 1 - 2 * pairs[:, 1]
Q = 1 - 2 * pairs[:, 0]

X = (I + 1j*Q) / np.sqrt(2)

# ---------------------------------
# 4. OFDM modulation (IFFT)
# ---------------------------------
x = np.fft.ifft(X)

# ---------------------------------
# 5. Add cyclic prefix
# ---------------------------------
cp = x[-CP_LENGTH:]
x_with_cp = np.concatenate((cp, x))

# ---------------------------------
# 6. Multipath channel
# ---------------------------------
h = np.array([1, 0.9, 0.7, 0.5])

received = np.convolve(x_with_cp, h)

# ---------------------------------
# 7. Remove cyclic prefix
# ---------------------------------
y = received[CP_LENGTH:CP_LENGTH + N]

# ---------------------------------
# 8. OFDM demodulation (FFT)
# ---------------------------------
Y = np.fft.fft(y)

# ---------------------------------
# 9. Channel frequency response
# ---------------------------------
H = np.fft.fft(h, n=N)

# ---------------------------------
# 10. Zero-forcing equalization
# ---------------------------------
X_hat = Y / H

# ---------------------------------
# 11. QPSK detection
# ---------------------------------
detected_bit_0 = (X_hat.imag < 0).astype(int)
detected_bit_1 = (X_hat.real < 0).astype(int)

detected_pairs = np.column_stack(
    (detected_bit_0, detected_bit_1)
)

detected_bits = detected_pairs.reshape(-1)

# ---------------------------------
# 12. Results
# ---------------------------------
bit_errors = np.sum(bits != detected_bits)

print("Number of subcarriers:", N)
print("Cyclic prefix length:", CP_LENGTH)
print("Channel impulse response:", h)

print("Transmitted samples:", len(x_with_cp))
print("Number of bit errors:", bit_errors)

print(
    "Maximum equalization error:",
    np.max(np.abs(X - X_hat))
)


# ---------------------------------
# 13. Experiment: Remove the CP
# ---------------------------------

# Transmit without cyclic prefix
received_no_cp = np.convolve(x, h)

# Keep the first N received samples
y_no_cp = received_no_cp[:N]

# FFT
Y_no_cp = np.fft.fft(y_no_cp)

# Apply the SAME zero-forcing equalizer
X_hat_no_cp = Y_no_cp / H

# QPSK detection
detected_bit_0 = (X_hat_no_cp.imag < 0).astype(int)
detected_bit_1 = (X_hat_no_cp.real < 0).astype(int)

detected_pairs = np.column_stack(
    (detected_bit_0, detected_bit_1)
)

detected_bits_no_cp = detected_pairs.reshape(-1)

# Compare results
bit_errors_no_cp = np.sum(bits != detected_bits_no_cp)

print("\n--- CP Comparison ---")
print("With CP:   ", bit_errors, "bit errors")
print("Without CP:", bit_errors_no_cp, "bit errors")

print(
    "Equalization error without CP:",
    np.max(np.abs(X - X_hat_no_cp))
)
