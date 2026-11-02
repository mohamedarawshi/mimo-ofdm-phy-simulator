
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Simulation parameters
N_BITS = 1_000_000
rng = np.random.default_rng(42)

EbN0_dB_values = np.arange(0, 11, 1)

# Generate random bits
bits = rng.integers(0, 2, size=N_BITS)
bit_pairs = bits.reshape(-1, 2)

# QPSK mapper
I = 1 - 2 * bit_pairs[:, 1]
Q = 1 - 2 * bit_pairs[:, 0]
symbols = (I + 1j * Q) / np.sqrt(2)

simulated_BER = []
theoretical_BER = []

for EbN0_dB in EbN0_dB_values:

    EbN0 = 10 ** (EbN0_dB / 10)

    Es = 1
    Eb = Es / 2
    N0 = Eb / EbN0

    # AWGN channel
    noise = np.sqrt(N0 / 2) * (
        rng.standard_normal(len(symbols))
        + 1j * rng.standard_normal(len(symbols))
    )

    received = symbols + noise

    # ML detector
    detected_bit_0 = (received.imag < 0).astype(int)
    detected_bit_1 = (received.real < 0).astype(int)

    detected_pairs = np.column_stack(
        (detected_bit_0, detected_bit_1)
    )
    detected_bits = detected_pairs.reshape(-1)

    # Simulated BER
    bit_errors = np.sum(bits != detected_bits)
    BER = bit_errors / N_BITS
    simulated_BER.append(BER)

    # Theoretical QPSK BER
    theoretical = 0.5 * erfc(np.sqrt(EbN0))
    theoretical_BER.append(theoretical)

    print(f"Eb/N0 = {EbN0_dB:2d} dB | BER = {BER:.8f}")

# Plot
plt.figure(figsize=(9, 6))

plt.semilogy(
    EbN0_dB_values,
    simulated_BER,
    "o-",
    label="Simulated BER"
)

plt.semilogy(
    EbN0_dB_values,
    theoretical_BER,
    "--",
    label="Theoretical BER"
)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")
plt.title("QPSK over AWGN: Simulated vs Theoretical BER")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()

plt.savefig("qpsk_ber.png", dpi=200)
plt.show()
