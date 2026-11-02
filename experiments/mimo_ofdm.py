
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ==========================================
# 1. Simulation parameters
# ==========================================

N = 64
CP = 16
NUM_OFDM = 2000

NUM_TX = 2
NUM_RX = 2

Es = 1.0
Eb = Es / 2

EbN0_dB_values = np.arange(0, 21, 2)

# ==========================================
# 2. Four multipath channels
# ==========================================

# h[rx, tx, delay]
# Each channel has 4 taps.
#
# The two spatial paths are deliberately
# correlated, making equalization challenging.

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

# H_freq[rx, tx, k]
H_freq = np.fft.fft(h, n=N, axis=2)

# Rearrange to H[k, rx, tx]
H_freq = np.moveaxis(H_freq, 2, 0)

# ==========================================
# 3. QPSK modulation / detection
# ==========================================

def qpsk_modulate(bits):
    pairs = bits.reshape(-1, 2)

    I = 1 - 2 * pairs[:, 1]
    Q = 1 - 2 * pairs[:, 0]

    return (I + 1j * Q) / np.sqrt(2)


def qpsk_detect(symbols):
    bits = np.empty(2 * len(symbols), dtype=int)

    bits[0::2] = (symbols.imag < 0).astype(int)
    bits[1::2] = (symbols.real < 0).astype(int)

    return bits


# ==========================================
# 4. Generate transmitted data
# ==========================================

# Shape: [TX antenna, OFDM symbol, bits]
bits = rng.integers(
    0, 2,
    size=(NUM_TX, NUM_OFDM, 2 * N)
)

# QPSK symbols: [TX, OFDM, subcarrier]
X = qpsk_modulate(bits).reshape(
    NUM_TX, NUM_OFDM, N
)

# ==========================================
# 5. OFDM modulation
# ==========================================

# IFFT across the subcarrier axis
x = np.fft.ifft(X, axis=2)

# Add cyclic prefix
x_cp = np.concatenate(
    (x[:, :, -CP:], x),
    axis=2
)

# ==========================================
# 6. Multipath MIMO channel
# ==========================================

# Continuous transmission:
# concatenate all OFDM symbols for each TX
tx_streams = x_cp.reshape(NUM_TX, -1)

rx_length = tx_streams.shape[1]

rx_clean = np.zeros(
    (NUM_RX, rx_length),
    dtype=complex
)

for rx in range(NUM_RX):
    for tx in range(NUM_TX):

        # Linear convolution through each path
        contribution = np.convolve(
            tx_streams[tx],
            h[rx, tx]
        )

        # Keep the observation window
        rx_clean[rx] += contribution[:rx_length]

# ==========================================
# 7. BER versus Eb/N0
# ==========================================

ber_zf = []
ber_mmse = []

mse_zf = []
mse_mmse = []

total_bits = bits.size

for EbN0_dB in EbN0_dB_values:

    EbN0 = 10 ** (EbN0_dB / 10)
    N0 = Eb / EbN0

    # Time-domain noise variance.
    # With NumPy FFT conventions, the
    # frequency-domain variance becomes N0.
    noise_var = N0 / N

    noise = np.sqrt(noise_var / 2) * (
        rng.standard_normal(rx_clean.shape)
        + 1j * rng.standard_normal(rx_clean.shape)
    )

    rx_noisy = rx_clean + noise

    # ======================================
    # 8. OFDM demodulation
    # ======================================

    # [RX, OFDM, N+CP]
    rx_blocks = rx_noisy.reshape(
        NUM_RX, NUM_OFDM, N + CP
    )

    # Remove CP
    y = rx_blocks[:, :, CP:CP + N]

    # FFT
    Y = np.fft.fft(y, axis=2)

    # Rearrange:
    # Y[k, rx, ofdm_symbol]
    Y_k = np.transpose(Y, (2, 0, 1))

    # ======================================
    # 9. ZF MIMO equalization
    # ======================================

    # Solve H[k] X_hat[k] = Y[k]
    # for every subcarrier k.
    X_zf_k = np.linalg.solve(
        H_freq,
        Y_k
    )

    # ======================================
    # 10. MMSE MIMO equalization
    # ======================================

    H_hermitian = np.conjugate(
        np.swapaxes(H_freq, 1, 2)
    )

    HhH = H_hermitian @ H_freq

    regularization = (
        (N0 / Es) * np.eye(NUM_TX)
    )

    G_mmse = np.linalg.solve(
        HhH + regularization,
        H_hermitian
    )

    X_mmse_k = G_mmse @ Y_k

    # ======================================
    # 11. Restore original data dimensions
    # ======================================

    # [TX, OFDM, subcarrier]
    X_zf = np.transpose(
        X_zf_k, (1, 2, 0)
    )

    X_mmse = np.transpose(
        X_mmse_k, (1, 2, 0)
    )

    # ======================================
    # 12. Calculate BER
    # ======================================

    detected_zf = qpsk_detect(
        X_zf.reshape(-1)
    )

    detected_mmse = qpsk_detect(
        X_mmse.reshape(-1)
    )

    original_bits = bits.reshape(-1)

    errors_zf = np.sum(
        detected_zf != original_bits
    )

    errors_mmse = np.sum(
        detected_mmse != original_bits
    )

    ber_zf.append(errors_zf / total_bits)
    ber_mmse.append(errors_mmse / total_bits)

    # ======================================
    # 13. Calculate symbol MSE
    # ======================================

    mse_zf.append(
        np.mean(np.abs(X_zf - X)**2)
    )

    mse_mmse.append(
        np.mean(np.abs(X_mmse - X)**2)
    )

    print(
        f"Eb/N0={EbN0_dB:2d} dB | "
        f"ZF BER={ber_zf[-1]:.6f} | "
        f"MMSE BER={ber_mmse[-1]:.6f} | "
        f"ZF MSE={mse_zf[-1]:.4f} | "
        f"MMSE MSE={mse_mmse[-1]:.4f}"
    )

# ==========================================
# 14. BER plot
# ==========================================

plt.figure(figsize=(10, 6))

plt.semilogy(
    EbN0_dB_values,
    ber_zf,
    "o-",
    label="ZF"
)

plt.semilogy(
    EbN0_dB_values,
    ber_mmse,
    "s-",
    label="MMSE"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate")
plt.title("2x2 MIMO-OFDM: ZF vs MMSE BER")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig("mimo_ofdm_ber.png", dpi=200)
plt.show()

# ==========================================
# 15. MSE plot
# ==========================================

plt.figure(figsize=(10, 6))

plt.semilogy(
    EbN0_dB_values,
    mse_zf,
    "o-",
    label="ZF"
)

plt.semilogy(
    EbN0_dB_values,
    mse_mmse,
    "s-",
    label="MMSE"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Symbol MSE")
plt.title("2x2 MIMO-OFDM: ZF vs MMSE MSE")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig("mimo_ofdm_mse.png", dpi=200)
plt.show()


# ==========================================
# 16. Channel condition number analysis
# ==========================================

condition_numbers = np.linalg.cond(H_freq)

worst_k = np.argmax(condition_numbers)

print("\n--- Channel Conditioning ---")
print("Worst subcarrier:", worst_k)
print("Maximum condition number:", condition_numbers[worst_k])

print("\nSingular values at worst subcarrier:")
print(np.linalg.svd(H_freq[worst_k], compute_uv=False))

plt.figure(figsize=(10, 6))

plt.semilogy(
    np.arange(N),
    condition_numbers,
    "o-",
    markersize=4
)

plt.axvline(
    worst_k,
    linestyle="--",
    label=f"Worst subcarrier: {worst_k}"
)

plt.xlabel("Subcarrier index")
plt.ylabel("Condition number")
plt.title("2x2 MIMO-OFDM Channel Conditioning")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig("mimo_condition_numbers.png", dpi=200)
plt.show()
