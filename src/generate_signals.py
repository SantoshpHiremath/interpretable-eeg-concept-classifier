"""Synthetic EEG-shaped multi-channel signal generator.

Two classes differ in BOTH a time-domain pattern (a localized waveform
template appearing at a class-dependent time window) and a frequency-domain
pattern (a class-dependent dominant oscillation band) -- deliberately, so a
model using only one domain cannot solve the full task. Amplitudes are kept
close to the background noise level so the task is not trivially easy (see
README "honest finding" for the bug this fixes).

Shape mirrors real EEG: multi-channel, fixed sample rate, short epochs.
"""
from __future__ import annotations

import numpy as np
import torch

N_CHANNELS = 8
N_TIMESTEPS = 256
SAMPLE_RATE_HZ = 128.0

# Class 0: waveform template appears early (t in [30, 60)); dominant band ~10 Hz.
# Class 1: waveform template appears late (t in [180, 210)); dominant band ~21 Hz.
CLASS_TIME_WINDOWS = {0: (30, 60), 1: (180, 210)}
CLASS_DOMINANT_HZ = {0: 10.0, 1: 21.0}

TEMPLATE_AMPLITUDE = 0.75
OSCILLATION_AMPLITUDE = 0.28
NOISE_STD = 0.4


def _time_axis(n_timesteps: int = N_TIMESTEPS, sample_rate_hz: float = SAMPLE_RATE_HZ) -> np.ndarray:
    return np.arange(n_timesteps) / sample_rate_hz


def _waveform_template(length: int = 30) -> np.ndarray:
    """A localized Gabor-like burst -- a real, recognizable temporal 'shape',
    not just a spike, so time-concept localization has something structured
    to match against.
    """
    t = np.linspace(-1.5, 1.5, length)
    return np.exp(-t**2) * np.sin(2 * np.pi * 3.0 * t)


TIME_JITTER_MAX = 10  # timesteps the template's onset can shift, +/-


def _inject_time_pattern(signal: np.ndarray, window: tuple, rng: np.random.Generator,
                          amplitude: float = TEMPLATE_AMPLITUDE) -> np.ndarray:
    start, end = window
    length = end - start
    # Onset jitter: real signals aren't perfectly time-locked to a fixed
    # window, and without this the model can memorize an exact position
    # instead of learning a genuine temporal pattern -- see README.
    shift = int(rng.integers(-TIME_JITTER_MAX, TIME_JITTER_MAX + 1))
    start = max(0, min(len(signal) - length, start + shift))
    end = start + length
    template = _waveform_template(length) * amplitude
    amp_jitter = rng.normal(1.0, 0.08)  # per-channel amplitude jitter
    signal[start:end] += template * amp_jitter
    return signal


def _inject_frequency_pattern(signal: np.ndarray, dominant_hz: float, rng: np.random.Generator,
                               amplitude: float = OSCILLATION_AMPLITUDE) -> np.ndarray:
    t = _time_axis()
    jitter = rng.normal(1.0, 0.08)
    signal += amplitude * jitter * np.sin(2 * np.pi * dominant_hz * t)
    return signal


def generate_dataset(n_samples: int = 400, seed: int = 42, mode: str = "dual") -> tuple:
    """mode: 'dual' (both domains distinguish the classes -- the main dataset),
    'time_only' (only the time-domain pattern differs; frequency content is
    class-independent noise), or 'freq_only' (only frequency content differs;
    no localized time-domain template at all). Used by the ablation tests to
    prove the dual-domain architecture is doing real work, not being decorative.
    """
    assert mode in ("dual", "time_only", "freq_only")
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, N_CHANNELS, N_TIMESTEPS), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        label = i % 2
        y[i] = label
        for ch in range(N_CHANNELS):
            signal = rng.normal(0.0, NOISE_STD, size=N_TIMESTEPS)
            if mode in ("dual", "time_only"):
                signal = _inject_time_pattern(signal, CLASS_TIME_WINDOWS[label], rng)
            if mode in ("dual", "freq_only"):
                signal = _inject_frequency_pattern(signal, CLASS_DOMINANT_HZ[label], rng)
            X[i, ch, :] = signal

    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]
    return torch.from_numpy(X), torch.from_numpy(y)


def train_val_test_split(X: torch.Tensor, y: torch.Tensor, train_frac: float = 0.6,
                          val_frac: float = 0.2, seed: int = 0) -> tuple:
    """Stratified split: each class is split into train/val/test in the same
    proportions independently, then concatenated and shuffled. A plain
    sequential slice of a randomly-permuted dataset is NOT guaranteed to be
    class-balanced in the smaller val/test slices -- this was a real bug
    found during development (see README): a skewed val split (54/26,
    ~67%/33%) made an untrained model's val accuracy look like a real ~0.68
    signal (it was the majority-class baseline), which made a genuinely
    struggling model's accuracy look like it was "degrading during training"
    when what was really happening was the model moving away from a
    deceptively strong majority-class baseline. Every accuracy number in this
    project's tests and pipeline is computed on a stratified split.
    """
    rng = np.random.default_rng(seed)
    y_np = y.numpy()
    train_idx, val_idx, test_idx = [], [], []

    for label in np.unique(y_np):
        idx = np.where(y_np == label)[0]
        idx = rng.permutation(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])

    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    test_idx = rng.permutation(test_idx)

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
        (X[test_idx], y[test_idx]),
    )
