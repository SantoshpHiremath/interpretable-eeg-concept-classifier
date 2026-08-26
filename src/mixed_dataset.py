"""A dataset requiring BOTH time and frequency cues to solve fully: half the
samples are separable only by the time-domain template (frequency content is
class-independent noise for those samples), half are separable only by the
frequency-domain oscillation (time content is class-independent noise for
those samples). Used by the dual-domain ablation test -- the core evidence
that combining both domains adds real capability, not just redundant capacity.
"""
from __future__ import annotations

import numpy as np
import torch

from src.generate_signals import (
    N_CHANNELS, N_TIMESTEPS, NOISE_STD, CLASS_TIME_WINDOWS, CLASS_DOMINANT_HZ,
    _inject_time_pattern, _inject_frequency_pattern,
)


def generate_mixed_dataset(n_samples: int = 400, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, N_CHANNELS, N_TIMESTEPS), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    half = n_samples // 2

    for i in range(n_samples):
        label = i % 2
        y[i] = label
        use_time_cue = i < half
        for ch in range(N_CHANNELS):
            signal = rng.normal(0.0, NOISE_STD, size=N_TIMESTEPS)
            if use_time_cue:
                signal = _inject_time_pattern(signal, CLASS_TIME_WINDOWS[label], rng)
            else:
                signal = _inject_frequency_pattern(signal, CLASS_DOMINANT_HZ[label], rng)
            X[i, ch, :] = signal

    perm = rng.permutation(n_samples)
    X, y = X[perm], y[perm]
    return torch.from_numpy(X), torch.from_numpy(y)
