
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ==========================================
# 1. OFDM and MIMO parameters
# ==========================================

N = 64
CP = 16
NUM_TX = 2
NUM_RX = 2
EbN0_dB = 10

# Four multipath channels: [RX, TX, tap]
h = np.array([
    [
        [1.0, 0.4, 0.2, 0.1],
        [0.9, 0.35, 0.15, 0.05]
    ],
    [
        [0.9, 0.3, 0.1, 0.05],
        [1.0, 0.45, 0.2, 0.1]
    ]
], dtype=complex)

L = h.shape[2]

assert CP >= L - 1

# True channel is used ONLY for evaluation.
H_true = np.fft.fft(h, n=N, axis=2)

# ==========================================
# 2. Orthogonal pilot design
# ==========================================

# Rows: transmit antennas
# Columns: consecutive pilot OFDM symbols
P = np.array([
    [1, 1],
    [1, -1]
], dtype=complex)

print("Pilot Gram matrix:")
print(P @ P.conj().T)

# Repeat pilots on every subcarrier.
# Shape: [TX, pilot OFDM symbol, subcarrier]
X_pilot = np.repeat(P[:, :, None], N, axis=2)

# ==========================================
# 3. OFDM modulation
# ==========================================

# Convert pilots to time domain.
x = np.fft.ifft(X_pilot, axis=2)

# Add cyclic prefix.
x_cp = np.concatenate(
    [x[:, :, -CP:], x],
    axis=2
)

# Serialize the two pilot OFDM symbols.
tx_streams = x_cp.reshape(NUM_TX, -1)

# ==========================================
# 4. Four multipath channels
# ==========================================

rx_length = tx_streams.shape[1]

rx_clean = np.zeros(
    (NUM_RX, rx_length),
    dtype=complex
)

for rx in range(NUM_RX):
    for tx in range(NUM_TX):

        contribution = np.convolve(
            tx_streams[tx],
            h[rx, tx]
        )

        rx_clean[rx] += contribution[:rx_length]

# ==========================================
# 5. Add AWGN
# ==========================================

EbN0 = 10 ** (EbN0_dB / 10)

Es = 1.0
Eb = Es / 2
N0 = Eb / EbN0

# Frequency-domain noise variance is N0.
# With NumPy FFT, time-domain variance is N0/N.
noise_var = N0 / N

noise = np.sqrt(noise_var / 2) * (
    rng.standard_normal(rx_clean.shape)
    + 1j * rng.standard_normal(rx_clean.shape)
)

rx_noisy = rx_clean + noise

# ==========================================
# 6. Receiver: CP removal and FFT
# ==========================================

rx_blocks = rx_noisy.reshape(
    NUM_RX, 2, N + CP
)

# Remove cyclic prefix.
y = rx_blocks[:, :, CP:CP + N]

# FFT to frequency domain.
Y = np.fft.fft(y, axis=2)

# ==========================================
# 7. LS channel estimation
# ==========================================

# For each subcarrier:
#
# Y_p[k] = H[k] P + W[k]
#
# LS estimator:
#
# H_LS[k] = Y_p[k] P^H (P P^H)^(-1)
#
# Since P P^H = 2I:
#
# H_LS[k] = (1/2) Y_p[k] P^H

H_est = np.zeros(
    (NUM_RX, NUM_TX, N),
    dtype=complex
)

for k in range(N):

    H_est[:, :, k] = (
        Y[:, :, k] @ P.conj().T / 2
    )

# ==========================================
# 8. DFT-based channel estimation
# ==========================================

# Convert all four estimated channel
# responses to the delay domain.
h_est = np.fft.ifft(H_est, axis=2)

# The physical channel has only L taps.
# Discard the remaining noisy taps.
h_est[:, :, L:] = 0

# Convert back to frequency domain.
H_dft = np.fft.fft(h_est, axis=2)

# ==========================================
# 9. Calculate channel estimation MSE
# ==========================================

mse_ls = np.mean(
    np.abs(H_est - H_true)**2
)

mse_dft = np.mean(
    np.abs(H_dft - H_true)**2
)

# Each pilot has unit energy.
# Orthogonal pilots give LS error variance N0/2.
theory_ls = N0 / 2

# Retaining L of N delay taps reduces
# the expected noise energy by L/N.
theory_dft = theory_ls * L / N

print(f"\nEb/N0 = {EbN0_dB} dB")

print("\n--- Channel Estimation Comparison ---")

print(f"LS MSE:              {mse_ls:.6f}")
print(f"DFT MSE:             {mse_dft:.6f}")
print(f"Theoretical LS MSE:  {theory_ls:.6f}")
print(f"Theoretical DFT MSE: {theory_dft:.6f}")
print(f"MSE improvement:     {mse_ls / mse_dft:.2f}x")

# ==========================================
# 10. Plot true, LS, and DFT estimates
# ==========================================

k_values = np.arange(N)

plt.figure(figsize=(10, 6))

plt.plot(
    k_values,
    np.abs(H_true[0, 0]),
    label="True H11",
    linewidth=2
)

plt.plot(
    k_values,
    np.abs(H_est[0, 0]),
    "--",
    label="LS estimate",
    alpha=0.7
)

plt.plot(
    k_values,
    np.abs(H_dft[0, 0]),
    "-.",
    label="DFT-based estimate",
    linewidth=2
)

plt.xlabel("Subcarrier index")
plt.ylabel("Channel magnitude")
plt.title("MIMO Channel Estimation: LS vs DFT-Based")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    "mimo_dft_channel_estimation.png",
    dpi=200
)

plt.show()

# ==========================================
# 11. Plot channel estimation error
# ==========================================

error_ls = np.abs(H_est[0, 0] - H_true[0, 0])**2
error_dft = np.abs(H_dft[0, 0] - H_true[0, 0])**2

plt.figure(figsize=(10, 6))

plt.semilogy(
    k_values,
    error_ls + 1e-15,
    label="LS squared error"
)

plt.semilogy(
    k_values,
    error_dft + 1e-15,
    label="DFT-based squared error"
)

plt.xlabel("Subcarrier index")
plt.ylabel("Squared channel estimation error")
plt.title("MIMO Channel Estimation Error per Subcarrier")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig(
    "mimo_channel_estimation_error.png",
    dpi=200
)

plt.show()
