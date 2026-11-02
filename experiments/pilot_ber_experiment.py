
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Simulation parameters
# ==========================================

rng = np.random.default_rng(42)

N = 64
CP = 16
NUM_OFDM = 2000

# True multipath channel
h = np.array([1, 0.9, 0.7, 0.5])
H_true = np.fft.fft(h, n=N)

# Pilot and data subcarriers
pilots = np.arange(0, N, 8)
data_idx = np.setdiff1d(np.arange(N), pilots)

num_bits = 2 * len(data_idx)

# SNR values
EbN0_dB_values = np.arange(0, 21, 2)

# BER results
ber_perfect = []
ber_ls = []
ber_dft = []

# ==========================================
# 2. DFT-based estimator setup
# ==========================================

# Assume the receiver knows the maximum
# channel length, but not the tap values.
L_est = 4

# Pilot observation matrix:
# A[p,l] = exp(-j*2*pi*k_p*l/N)
A = np.exp(
    -1j * 2 * np.pi
    * pilots[:, None]
    * np.arange(L_est)[None, :]
    / N
)

# ==========================================
# 3. QPSK detector
# ==========================================

def detect_qpsk(symbols):
    bits = np.empty(2 * len(symbols), dtype=int)

    bits[0::2] = (symbols.imag < 0).astype(int)
    bits[1::2] = (symbols.real < 0).astype(int)

    return bits


# ==========================================
# 4. BER simulation
# ==========================================

for EbN0_dB in EbN0_dB_values:

    EbN0 = 10 ** (EbN0_dB / 10)

    # Unit-energy QPSK: Eb = 1/2.
    # This normalization excludes CP and
    # pilot overhead from Eb.
    N0 = 1 / (2 * EbN0)
    noise_var = N0 / N

    errors_perfect = 0
    errors_ls = 0
    errors_dft = 0

    for _ in range(NUM_OFDM):

        # ----------------------------------
        # Transmitter: Generate QPSK data
        # ----------------------------------

        bits = rng.integers(0, 2, num_bits)

        pairs = bits.reshape(-1, 2)

        I = 1 - 2 * pairs[:, 1]
        Q = 1 - 2 * pairs[:, 0]

        data = (I + 1j * Q) / np.sqrt(2)

        # ----------------------------------
        # Insert pilots and data
        # ----------------------------------

        X = np.zeros(N, dtype=complex)

        X[data_idx] = data
        X[pilots] = 1 + 0j

        # ----------------------------------
        # OFDM modulation
        # ----------------------------------

        x = np.fft.ifft(X)

        x_cp = np.concatenate(
            (x[-CP:], x)
        )

        # ----------------------------------
        # Multipath channel + AWGN
        # ----------------------------------

        rx_clean = np.convolve(x_cp, h)

        noise = np.sqrt(noise_var / 2) * (
            rng.standard_normal(len(rx_clean))
            + 1j * rng.standard_normal(len(rx_clean))
        )

        rx = rx_clean + noise

        # ----------------------------------
        # OFDM receiver: CP removal + FFT
        # ----------------------------------

        y = rx[CP:CP + N]

        Y = np.fft.fft(y)

        # ==================================
        # Receiver 1: Perfect CSI
        # ==================================

        X_perfect = (
            Y[data_idx] / H_true[data_idx]
        )

        detected_perfect = detect_qpsk(X_perfect)

        errors_perfect += np.sum(
            detected_perfect != bits
        )

        # ==================================
        # Receiver 2: LS + interpolation
        # ==================================

        # LS estimates at pilot positions
        H_pilots = Y[pilots] / X[pilots]

        # Periodic linear interpolation
        extended_idx = np.append(pilots, N)

        extended_H = np.append(
            H_pilots,
            H_pilots[0]
        )

        H_est = (
            np.interp(
                np.arange(N),
                extended_idx,
                extended_H.real
            )
            + 1j * np.interp(
                np.arange(N),
                extended_idx,
                extended_H.imag
            )
        )

        # Zero-forcing equalization
        X_ls = (
            Y[data_idx] / H_est[data_idx]
        )

        detected_ls = detect_qpsk(X_ls)

        errors_ls += np.sum(
            detected_ls != bits
        )

        # ==================================
        # Receiver 3: DFT-Based LS
        # ==================================

        # Estimate the 4 channel taps:
        # h_hat = argmin ||A*h - H_pilots||^2
        h_est = np.linalg.lstsq(
            A, H_pilots, rcond=None
        )[0]

        # Reconstruct all 64 channel values
        H_dft = np.fft.fft(h_est, n=N)

        # Zero-forcing equalization
        X_dft = (
            Y[data_idx] / H_dft[data_idx]
        )

        detected_dft = detect_qpsk(X_dft)

        errors_dft += np.sum(
            detected_dft != bits
        )

    # ======================================
    # BER calculation for this SNR
    # ======================================

    total_bits = NUM_OFDM * num_bits

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
        f"Linear LS={ber_ls[-1]:.6f} | "
        f"DFT LS={ber_dft[-1]:.6f}"
    )


# ==========================================
# 5. Plot BER comparison
# ==========================================

plt.figure(figsize=(10, 6))

# Replace zero measured BER with NaN for
# plotting only. Actual BER results stay zero.
def plot_ber(values):
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, np.nan)


plt.semilogy(
    EbN0_dB_values,
    plot_ber(ber_perfect),
    "o-",
    label="Perfect CSI",
    linewidth=2
)

plt.semilogy(
    EbN0_dB_values,
    plot_ber(ber_ls),
    "s-",
    label="LS + Linear Interpolation",
    linewidth=2
)

plt.semilogy(
    EbN0_dB_values,
    plot_ber(ber_dft),
    "^-",
    label="DFT-Based LS (4 taps)",
    linewidth=2
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")

plt.title(
    "OFDM BER: Perfect CSI vs Channel Estimation"
)

plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig(
    "pilot_ber_comparison.png",
    dpi=200
)

plt.show()
