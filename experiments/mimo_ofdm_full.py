
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ==========================================
# 1. Simulation parameters
# ==========================================

N = 64
CP = 16

NUM_TX = 2
NUM_RX = 2

NUM_DATA_OFDM = 2000
NUM_PILOT_OFDM = 2

Es = 1.0
Eb = Es / 2

EbN0_dB_values = np.arange(0, 21, 2)

# ==========================================
# 2. Four multipath channels
# ==========================================

# Shape: [RX, TX, delay tap]
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

# True channel: evaluation / ideal baseline only
H_true = np.fft.fft(h, n=N, axis=2)

# Shape: [subcarrier, RX, TX]
H_true_k = np.moveaxis(H_true, 2, 0)

# ==========================================
# 3. QPSK functions
# ==========================================

def qpsk_modulate(bits):
    pairs = bits.reshape(-1, 2)

    I = 1 - 2 * pairs[:, 1]
    Q = 1 - 2 * pairs[:, 0]

    return (I + 1j * Q) / np.sqrt(2)


def qpsk_detect(symbols):
    flat = np.asarray(symbols).reshape(-1)

    bits = np.empty(2 * flat.size, dtype=int)

    bits[0::2] = (flat.imag < 0).astype(int)
    bits[1::2] = (flat.real < 0).astype(int)

    return bits

# ==========================================
# 4. Generate orthogonal pilots
# ==========================================

# Rows = transmit antennas
# Columns = pilot OFDM symbols
P = np.array([
    [1, 1],
    [1, -1]
], dtype=complex)

assert np.allclose(
    P @ P.conj().T,
    2 * np.eye(NUM_TX)
)

# [TX, pilot symbol, subcarrier]
X_pilot = np.repeat(
    P[:, :, None],
    N,
    axis=2
)

# ==========================================
# 5. Generate QPSK data
# ==========================================

bits = rng.integers(
    0, 2,
    size=(NUM_TX, NUM_DATA_OFDM, 2 * N)
)

X_data = qpsk_modulate(bits).reshape(
    NUM_TX,
    NUM_DATA_OFDM,
    N
)

original_bits = bits.reshape(-1)
total_bits = original_bits.size

# ==========================================
# 6. Combine pilots and data
# ==========================================

# Two pilot symbols come first.
X_all = np.concatenate(
    [X_pilot, X_data],
    axis=1
)

NUM_TOTAL_OFDM = (
    NUM_PILOT_OFDM + NUM_DATA_OFDM
)

# ==========================================
# 7. OFDM modulation
# ==========================================

x = np.fft.ifft(X_all, axis=2)

x_cp = np.concatenate(
    [x[:, :, -CP:], x],
    axis=2
)

# Continuous waveform for each antenna
tx_streams = x_cp.reshape(NUM_TX, -1)

# ==========================================
# 8. MIMO multipath propagation
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
# 9. MMSE equalizer function
# ==========================================

def mmse_equalize(Y_data, H_est_k, N0):
    """
    Y_data shape: [RX, OFDM symbol, subcarrier]
    H_est_k shape: [subcarrier, RX, TX]

    Returns:
        X_hat shape: [TX, OFDM symbol, subcarrier]
    """

    # [k, RX, OFDM symbol]
    Y_k = np.transpose(
        Y_data,
        (2, 0, 1)
    )

    # Hermitian transpose of each H[k]
    Hh = np.conjugate(
        np.swapaxes(H_est_k, 1, 2)
    )

    # H^H H
    HhH = Hh @ H_est_k

    # MMSE regularization
    A = HhH + (N0 / Es) * np.eye(NUM_TX)

    # Solve A X_hat = H^H Y
    rhs = Hh @ Y_k

    X_hat_k = np.linalg.solve(
        A,
        rhs
    )

    # Restore [TX, OFDM, subcarrier]
    X_hat = np.transpose(
        X_hat_k,
        (1, 2, 0)
    )

    return X_hat

# ==========================================
# 10. BER experiment
# ==========================================

ber_perfect = []
ber_ls = []
ber_dft = []

channel_mse_ls = []
channel_mse_dft = []

for EbN0_dB in EbN0_dB_values:

    EbN0 = 10 ** (EbN0_dB / 10)
    N0 = Eb / EbN0

    # With NumPy FFT convention:
    # FFT-domain noise variance = N0
    noise_var = N0 / N

    noise = np.sqrt(noise_var / 2) * (
        rng.standard_normal(rx_clean.shape)
        + 1j * rng.standard_normal(rx_clean.shape)
    )

    rx_noisy = rx_clean + noise

    # ======================================
    # 11. Receiver FFT
    # ======================================

    rx_blocks = rx_noisy.reshape(
        NUM_RX,
        NUM_TOTAL_OFDM,
        N + CP
    )

    y = rx_blocks[:, :, CP:CP + N]

    Y_all = np.fft.fft(y, axis=2)

    # Separate pilot and data observations
    Y_pilot = Y_all[:, :NUM_PILOT_OFDM, :]
    Y_data = Y_all[:, NUM_PILOT_OFDM:, :]

    # ======================================
    # 12. LS channel estimation
    # ======================================

    H_ls = np.zeros(
        (NUM_RX, NUM_TX, N),
        dtype=complex
    )

    for k in range(N):
        H_ls[:, :, k] = (
            Y_pilot[:, :, k] @ P.conj().T / 2
        )

    # ======================================
    # 13. DFT-based channel estimation
    # ======================================

    h_ls = np.fft.ifft(H_ls, axis=2)

    # Retain only first L delay taps
    h_ls[:, :, L:] = 0

    H_dft = np.fft.fft(h_ls, axis=2)

    # Channel estimation MSE
    mse_ls = np.mean(
        np.abs(H_ls - H_true)**2
    )

    mse_dft = np.mean(
        np.abs(H_dft - H_true)**2
    )

    channel_mse_ls.append(mse_ls)
    channel_mse_dft.append(mse_dft)

    # ======================================
    # 14. Prepare estimated channel matrices
    # ======================================

    H_ls_k = np.moveaxis(H_ls, 2, 0)
    H_dft_k = np.moveaxis(H_dft, 2, 0)

    # ======================================
    # 15. Three MMSE receivers
    # ======================================

    X_hat_perfect = mmse_equalize(
        Y_data,
        H_true_k,
        N0
    )

    X_hat_ls = mmse_equalize(
        Y_data,
        H_ls_k,
        N0
    )

    X_hat_dft = mmse_equalize(
        Y_data,
        H_dft_k,
        N0
    )

    # ======================================
    # 16. QPSK detection and BER
    # ======================================

    detected_perfect = qpsk_detect(
        X_hat_perfect
    )

    detected_ls = qpsk_detect(
        X_hat_ls
    )

    detected_dft = qpsk_detect(
        X_hat_dft
    )

    errors_perfect = np.count_nonzero(
        detected_perfect != original_bits
    )

    errors_ls = np.count_nonzero(
        detected_ls != original_bits
    )

    errors_dft = np.count_nonzero(
        detected_dft != original_bits
    )

    ber_perfect.append(
        errors_perfect / total_bits
    )

    ber_ls.append(
        errors_ls / total_bits
    )

    ber_dft.append(
        errors_dft / total_bits
    )

    print(
        f"Eb/N0={EbN0_dB:2d} dB | "
        f"Perfect={ber_perfect[-1]:.6f} | "
        f"LS={ber_ls[-1]:.6f} | "
        f"DFT={ber_dft[-1]:.6f}"
    )

# ==========================================
# 17. Plot BER comparison
# ==========================================

plt.figure(figsize=(10, 6))

plt.semilogy(
    EbN0_dB_values,
    ber_perfect,
    "o-",
    label="Perfect CSI + MMSE"
)

plt.semilogy(
    EbN0_dB_values,
    ber_ls,
    "s-",
    label="LS + MMSE"
)

plt.semilogy(
    EbN0_dB_values,
    ber_dft,
    "^-",
    label="DFT-based LS + MMSE"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")
plt.title("2x2 MIMO-OFDM: Channel Estimation Impact on BER")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig(
    "mimo_ofdm_full_ber.png",
    dpi=200
)

plt.show()

# ==========================================
# 18. Plot channel estimation MSE
# ==========================================

plt.figure(figsize=(10, 6))

plt.semilogy(
    EbN0_dB_values,
    channel_mse_ls,
    "s-",
    label="LS estimation"
)

plt.semilogy(
    EbN0_dB_values,
    channel_mse_dft,
    "^-",
    label="DFT-based estimation"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Channel estimation MSE")
plt.title("2x2 MIMO-OFDM: Channel Estimation Accuracy")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig(
    "mimo_ofdm_full_channel_mse.png",
    dpi=200
)

plt.show()
