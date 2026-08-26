"""Time- and frequency-domain concept banks.

Each concept is a learned, inspectable prototype whose activation is
traceable back to a real location in the original signal (a timestep range
for time-concepts, a frequency band for frequency-concepts) -- this is what
makes them usable as an explanation, not just an internal embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FREQ_BANDS = 8
MAX_FREQ_HZ = 64.0  # Nyquist for 128 Hz sample rate


class TimeConceptBank(nn.Module):
    """A bank of learned 1D-convolutional concept prototypes. Each concept's
    activation is a SOFT (log-sum-exp) pooled cross-correlation between the
    prototype and the input signal, and the argmax position (a hard max,
    used only for reporting WHERE the concept matched, not for the
    activation value itself) is what explain.py uses to localize the concept
    to a real timestep range.

    Two design choices exist specifically because their absence caused a
    real, measured training failure during development (see README):

    1. Soft pooling, not hard max, for the activation value's gradient path.
       A hard max only back-propagates through the single best-matching
       position per sample per step -- a weak, high-variance signal in a
       low-data regime. Log-sum-exp pooling (at a high enough temperature to
       stay close to the true max -- see the `pooling_temperature` note
       below) lets every position contribute gradient proportional to how
       close it is to the max.

    2. ONE FILTER SHARED ACROSS CHANNELS (via grouped convolution, weight
       tied across the channel dimension), not one independent filter per
       channel. The data-generating process injects the SAME template shape
       on every channel (with only small per-channel amplitude jitter) plus
       INDEPENDENT per-channel noise -- so a per-channel-independent filter
       (the first version of this class, `nn.Conv1d(n_channels, n_concepts,
       kernel_size)`) has 8x more parameters than the task's real structure
       needs, and empirically it overfit: it reached 100% train accuracy but
       stayed at chance-level (~45-55%) validation accuracy across 4 seeds,
       even after fixing the pooling and normalization issues above and even
       though an oracle matched filter using the TRUE template achieves ~90%
       on the same held-out data (proving the task itself is learnable). A
       single filter correlated against each channel and averaged directly
       mirrors the true generative structure and cut the parameter count for
       this bank by 8x (from n_concepts*n_channels*kernel_size to
       n_concepts*kernel_size), which was the change that actually closed
       the train/validation gap.
    """

    def __init__(self, n_channels: int, n_concepts: int = 6, kernel_size: int = 30,
                 pooling_temperature: float = 40.0):
        super().__init__()
        self.n_channels = n_channels
        self.n_concepts = n_concepts
        self.kernel_size = kernel_size
        self.pooling_temperature = pooling_temperature
        # One prototype filter per concept, SHARED across channels: applied
        # independently to each channel (groups=n_channels would need
        # in_channels==out_channels, so instead we apply a single-input-
        # channel conv per concept and broadcast it across channels manually
        # for clarity and to keep the weight-sharing explicit and testable).
        self.prototypes = nn.Conv1d(in_channels=1, out_channels=n_concepts,
                                     kernel_size=kernel_size, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        """x: (batch, n_channels, n_timesteps). Returns activations (batch,
        n_concepts) = soft-pooled correlation per concept, and positions
        (batch, n_concepts) = the timestep where the HARD max occurred (for
        localization/explanation only -- not part of the activation's
        gradient path).
        """
        batch, n_channels, n_timesteps = x.shape
        x_flat = x.reshape(batch * n_channels, 1, n_timesteps)
        correlation_flat = self.prototypes(x_flat)  # (batch*n_channels, n_concepts, T-k+1)
        out_len = correlation_flat.shape[2]
        correlation = correlation_flat.reshape(batch, n_channels, self.n_concepts, out_len)
        correlation = correlation.mean(dim=1)  # average the SAME filter's response across channels

        # Soft (log-sum-exp) pooling for the trainable activation value.
        soft_activation = torch.logsumexp(correlation * self.pooling_temperature, dim=2) / self.pooling_temperature
        activations = F.relu(soft_activation)  # only positive matches count as "concept present"
        # Hard max used only to report a real timestep for explanations.
        with torch.no_grad():
            _, positions = correlation.max(dim=2)
        return {"activations": activations, "positions": positions}


class FrequencyConceptBank(nn.Module):
    """A bank of learned per-band attention weights over real power-spectral-
    density features (via torch.fft.rfft). Each concept's activation is a
    learned weighted combination of PSD bands, and the concept's "dominant
    band" (used for explanations) is the band with the highest attention
    weight for that concept.
    """

    def __init__(self, n_channels: int, n_concepts: int = 6, n_bands: int = N_FREQ_BANDS,
                 sample_rate_hz: float = 128.0):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_bands = n_bands
        self.sample_rate_hz = sample_rate_hz
        # Learned attention over (channel, band) -> concept.
        self.band_attention = nn.Parameter(torch.randn(n_concepts, n_channels, n_bands) * 0.1)

    def band_edges_hz(self, n_timesteps: int) -> torch.Tensor:
        nyquist = self.sample_rate_hz / 2.0
        return torch.linspace(0.0, nyquist, self.n_bands + 1)

    def power_spectral_density(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_channels, n_timesteps) -> (batch, n_channels, n_bands)
        real power spectral density, banded into n_bands equal-width bins.
        """
        batch, n_channels, n_timesteps = x.shape
        spectrum = torch.fft.rfft(x, dim=2)
        power = (spectrum.real ** 2 + spectrum.imag ** 2)  # (batch, n_channels, n_freqs)
        n_freqs = power.shape[2]

        edges = self.band_edges_hz(n_timesteps)
        freqs_hz = torch.linspace(0.0, self.sample_rate_hz / 2.0, n_freqs)

        bands = []
        for b in range(self.n_bands):
            lo, hi = edges[b].item(), edges[b + 1].item()
            mask = (freqs_hz >= lo) & (freqs_hz < hi if b < self.n_bands - 1 else freqs_hz <= hi)
            if mask.sum() == 0:
                bands.append(torch.zeros(batch, n_channels))
            else:
                bands.append(power[:, :, mask].mean(dim=2))
        return torch.stack(bands, dim=2)  # (batch, n_channels, n_bands)

    def forward(self, x: torch.Tensor) -> dict:
        psd = self.power_spectral_density(x)  # (batch, n_channels, n_bands)
        # activation[c] = sum over channels/bands of attention-weighted PSD
        weights = F.softmax(self.band_attention.reshape(self.n_concepts, -1), dim=1)
        weights = weights.reshape(self.n_concepts, self.band_attention.shape[1], self.n_bands)
        activations = torch.einsum("bcn,kcn->bk", psd, weights)
        return {"activations": activations, "psd": psd}

    def dominant_band_index(self, concept_idx: int) -> int:
        """The frequency band this concept attends to most, summed over channels
        -- used by explain.py to report a real Hz range for a concept.
        """
        weight_per_band = self.band_attention[concept_idx].sum(dim=0)
        return int(weight_per_band.argmax().item())
