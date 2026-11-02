
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ==========================================
# 1. Simulation parameters
# ==========================================

NUM_SYMBOLS = 100000
EbN0_dB_values = np.arange(0, 21, 2)

Es = 1.0
Eb = Es / 2

# Fixed 2x2 MIMO channel
H = np.array([
    [1.0, 0.9],
    [0.9, 1.0]
], dtype=complex)

ber_zf = []
ber_mmse = []

mse_zf = []
mse_mmse = []

# ==========================================
# 2. QPSK modulation / detection
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
# 3. Generate two independent QPSK streams
# ==========================================

bits1 = rng.integers(0, 2, 2 * NUM_SYMBOLS)
bits2 = rng.integers(0, 2, 2 * NUM_SYMBOLS)

x1 = qpsk_modulate(bits1)
x2 = qpsk_modulate(bits2)

# Each column represents one transmission time
X = np.vstack((x1, x2))

# ==========================================
# 4. Simulate BER versus Eb/N0
# ==========================================

for EbN0_dB in EbN0_dB_values:

    EbN0 = 10 ** (EbN0_dB / 10)
    N0 = Eb / EbN0

    # --------------------------------------
    # MIMO channel + AWGN
    # --------------------------------------

    W = np.sqrt(N0 / 2) * (
        rng.standard_normal(X.shape)
        + 1j * rng.standard_normal(X.shape)
    )

    Y = H @ X + W

    # ======================================
    # Receiver 1: Zero-Forcing
    # ======================================

    # Solve H * X_hat = Y
    X_zf = np.linalg.solve(H, Y)

    # ======================================
    # Receiver 2: MMSE
    # ======================================

    G_mmse = np.linalg.solve(
        H.conj().T @ H + (N0 / Es) * np.eye(2),
        H.conj().T
    )

    X_mmse = G_mmse @ Y

    # ======================================
    # BER calculation
    # ======================================

    bits_zf_1 = qpsk_detect(X_zf[0])
    bits_zf_2 = qpsk_detect(X_zf[1])

    bits_mmse_1 = qpsk_detect(X_mmse[0])
    bits_mmse_2 = qpsk_detect(X_mmse[1])

    total_bits = 4 * NUM_SYMBOLS

    errors_zf = (
        np.sum(bits_zf_1 != bits1)
        + np.sum(bits_zf_2 != bits2)
    )

    errors_mmse = (
        np.sum(bits_mmse_1 != bits1)
        + np.sum(bits_mmse_2 != bits2)
    )

    ber_zf.append(errors_zf / total_bits)
    ber_mmse.append(errors_mmse / total_bits)

    # ======================================
    # Symbol MSE calculation
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
# 5. Plot BER
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
plt.title("2x2 MIMO: ZF vs MMSE BER")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("mimo_ber.png", dpi=200)
plt.show()


# ==========================================
# 6. Plot symbol MSE
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
plt.title("2x2 MIMO: ZF vs MMSE Symbol MSE")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("mimo_mse.png", dpi=200)
plt.show()
