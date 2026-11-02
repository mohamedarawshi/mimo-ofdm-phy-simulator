import numpy as np
import pytest
from mimo_ofdm_phy import (qpsk_modulate, qpsk_detect, frequency_response,
    apply_mimo_fir, orthogonal_pilots, ls_channel_estimate, dft_denoise,
    mmse_equalize, zf_equalize)

def test_qpsk_roundtrip():
    bits = np.array([0,0, 0,1, 1,0, 1,1], dtype=np.uint8)
    assert np.array_equal(qpsk_detect(qpsk_modulate(bits)), bits)
    assert np.allclose(np.abs(qpsk_modulate(bits)), 1)

def test_orthogonal_pilots():
    p = orthogonal_pilots()
    assert np.allclose(p @ p.conj().T, 2 * np.eye(2))

def test_noiseless_ls_and_dft():
    rng = np.random.default_rng(5)
    n, length = 64, 4
    taps = (rng.normal(size=(2,2,length)) + 1j*rng.normal(size=(2,2,length))) / 4
    h = np.fft.fft(taps, n=n, axis=-1)
    p = orthogonal_pilots()
    y = np.einsum('rtk,tp->rpk', h, p)
    estimate = ls_channel_estimate(y, p)
    assert np.allclose(estimate, h, atol=1e-12)
    assert np.allclose(dft_denoise(estimate, length), h, atol=1e-12)

def test_mmse_noiseless_full_rank():
    rng = np.random.default_rng(6)
    h = np.array([[[1, .2],[.3,1.2]]] * 8, dtype=complex)
    x = (rng.normal(size=(2,5,8)) + 1j*rng.normal(size=(2,5,8)))
    y = np.einsum('krt,tmk->rmk', h, x)
    assert np.allclose(mmse_equalize(y, h, 0), x)
    assert np.allclose(zf_equalize(y, h), x)

def test_cyclic_prefix_linear_channel_matches_frequency_model():
    rng = np.random.default_rng(7)
    n, cp, l, blocks = 16, 4, 3, 3
    taps = rng.normal(size=(2,2,l)) + 1j*rng.normal(size=(2,2,l))
    x = rng.normal(size=(2,blocks,n)) + 1j*rng.normal(size=(2,blocks,n))
    time = np.fft.ifft(x, axis=-1)
    tx = np.concatenate([time[:,:,-cp:], time], axis=-1).reshape(2,-1)
    rx = apply_mimo_fir(tx, taps).reshape(2,blocks,n+cp)
    actual = np.fft.fft(rx[:,:,cp:], axis=-1)
    h = frequency_response(taps, n)
    expected = np.einsum('krt,tmk->rmk', h, x)
    assert np.allclose(actual, expected, atol=1e-12)

def test_dft_noise_projection_reduces_mse():
    rng = np.random.default_rng(8)
    n, l = 64, 4
    noise = (rng.normal(size=(2,2,n)) + 1j*rng.normal(size=(2,2,n))) / np.sqrt(2)
    filtered = dft_denoise(noise, l)
    assert np.mean(np.abs(filtered)**2) < 0.2 * np.mean(np.abs(noise)**2)

def test_invalid_pilot_rank():
    with pytest.raises(ValueError):
        ls_channel_estimate(np.ones((2,2,8)), np.ones((2,2)))
