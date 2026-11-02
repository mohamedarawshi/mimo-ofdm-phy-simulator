
import numpy as np

# Number of information bits
N_BITS = 1_000_000

# Reproducible random number generator
rng = np.random.default_rng(42)

# Generate random binary information
bits = rng.integers(0, 2, size=N_BITS)

print("Transmitted bits:", bits)


# Group the bits into pairs
bit_pairs = bits.reshape(-1, 2)

# Map each pair to a QPSK symbol
I = 1 - 2 * bit_pairs[:, 1]
Q = 1 - 2 * bit_pairs[:, 0]

symbols = (I + 1j * Q) / np.sqrt(2)

print("Bit pairs:")
print(bit_pairs)

print("QPSK symbols:")
print(symbols)


# AWGN channel
EbN0_dB = 0

EbN0 = 10 ** (EbN0_dB / 10)

Es = 1
Eb = Es / 2

N0 = Eb / EbN0

# Generate complex Gaussian noise
noise = np.sqrt(N0 / 2) * (
    rng.standard_normal(len(symbols))
    + 1j * rng.standard_normal(len(symbols))
)

# Received symbols
received = symbols + noise

print("Transmitted symbols:")
print(symbols)

print("Received symbols:")
print(received)


# ML detector for QPSK over AWGN

# Recover the first bit from the imaginary component
detected_bit_0 = (received.imag < 0).astype(int)

# Recover the second bit from the real component
detected_bit_1 = (received.real < 0).astype(int)

# Combine the detected bits into pairs
detected_pairs = np.column_stack(
    (detected_bit_0, detected_bit_1)
)

# Convert the pairs back into a sequence of bits
detected_bits = detected_pairs.reshape(-1)

print("Original bits: ", bits)
print("Detected bits: ", detected_bits)


# Count incorrect bits
bit_errors = np.sum(bits != detected_bits)

# Calculate bit error rate
BER = bit_errors / N_BITS

print("Number of bit errors:", bit_errors)
print("BER:", BER)
