"""The dual-domain self-explaining selective model (DualDomainSESM).

Self-explaining by construction: the classification head consumes ONLY the
concept activations (time + frequency), never the raw input directly. This
is checked structurally in tests/test_model.py, not just asserted here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.concepts import TimeConceptBank, FrequencyConceptBank


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Real sparsemax (Martins & Astudillo, 2016): the Euclidean projection of
    `logits` onto the probability simplex. Unlike a hard top-k/threshold mask
    (an earlier version of this function used mean-thresholding, which
    blocked gradient flow to every masked-out logit and collapsed training to
    a single always-on concept -- see README), sparsemax is piecewise-linear
    and differentiable almost everywhere, so every logit still receives a
    real gradient signal even on steps where its output is exactly zero.
    Produces genuinely sparse (exact-zero, not just small) outputs, which is
    the "selective" property SESM needs, without breaking training.
    """
    logits = logits - logits.max(dim=dim, keepdim=True).values  # numerical stability
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    n = logits.shape[dim]
    cumulative = sorted_logits.cumsum(dim=dim)
    k = torch.arange(1, n + 1, device=logits.device, dtype=logits.dtype)
    shape = [1] * logits.dim()
    shape[dim] = n
    k = k.reshape(shape)

    threshold_candidates = 1 + k * sorted_logits
    is_valid = threshold_candidates > cumulative
    k_max = is_valid.float().cumsum(dim=dim).max(dim=dim, keepdim=True).values
    k_max = k_max.clamp_min(1.0)

    cumulative_at_k = torch.gather(cumulative, dim, (k_max - 1).long())
    tau = (cumulative_at_k - 1) / k_max

    return torch.clamp(logits - tau, min=0.0)


class SelectiveHead(nn.Module):
    """Fuses time-concept and frequency-concept activations through a sparse
    gate, then a linear classifier over ONLY the gated (sparse) activations.
    """

    def __init__(self, n_time_concepts: int, n_freq_concepts: int, n_classes: int):
        super().__init__()
        self.n_time_concepts = n_time_concepts
        self.n_freq_concepts = n_freq_concepts
        # Per-domain rescale BEFORE fusion: time-concept activations (max
        # cross-correlation, roughly unit scale) and frequency-concept
        # activations (power-spectral-density-derived, roughly two orders of
        # magnitude larger) start on very different UNITS. Without any
        # rescaling, the gate and classifier are dominated by whichever
        # domain happens to have larger raw magnitude, regardless of which
        # domain is actually more predictive (confirmed: frequency
        # activations ~70x larger caused the gate to always select frequency
        # concepts and training to stall).
        #
        # This is a FIXED, buffered rescale computed once from a calibration
        # batch (see `calibrate_scale`), NOT BatchNorm/LayerNorm. Both of
        # those were tried and both actively broke learning on data where
        # only one domain carries real signal: by forcing every batch (or
        # every sample) to unit variance, they rescale a domain's pure NOISE
        # up to the same magnitude as the other domain's genuine signal, so
        # noise becomes just as "loud" as signal to the gate. Measured
        # effect: on a dataset where only the time domain was informative,
        # LayerNorm/BatchNorm-normalized models scored ~40-60% (chance-level)
        # across 4 seeds, and validation accuracy actively *degraded* while
        # training loss dropped -- the model was overfitting normalized noise
        # (see README). A fixed rescale corrects for units without
        # re-equalizing informativeness on every pass, so a domain that's
        # genuinely just noise for a given dataset stays low-amplitude
        # relative to the informative domain, exactly as it should.
        self.register_buffer("time_scale", torch.ones(1))
        self.register_buffer("freq_scale", torch.ones(1))
        self.calibrated = False
        self.gate_logits = nn.Linear(n_time_concepts + n_freq_concepts,
                                      n_time_concepts + n_freq_concepts)
        self.classifier = nn.Linear(n_time_concepts + n_freq_concepts, n_classes, bias=True)

    @torch.no_grad()
    def calibrate_scale(self, time_activations: torch.Tensor, freq_activations: torch.Tensor):
        """Sets a fixed per-domain rescale factor from a calibration batch, so
        both domains land in a comparable numeric range at the START of
        training -- a one-time unit conversion, not an ongoing per-batch/
        per-sample renormalization (see the class docstring for why an
        ongoing renormalization like BatchNorm/LayerNorm is wrong here).

        Rescales by mean absolute value: this equalizes the two domains'
        raw UNITS (correlation-based time activations vs. PSD-based
        frequency activations, which start ~70x larger in magnitude for
        reasons unrelated to informativeness) without also forcing their
        VARIANCE to match, which a std-based version of this calibration was
        tried and found to over-correct: it fixed the time-only-data case
        but made the frequency-only and dual-domain cases seed-sensitive
        (failing to train on roughly half of tested seeds), because
        variance genuinely differs between a smoothly-varying PSD-based
        signal and a sharp, sparse correlation-based one even when the
        domain IS informative. The residual seed-sensitivity that
        mean-magnitude calibration doesn't fully resolve (the time-domain
        branch is still a harder optimization than the frequency branch, in
        the sense that some random initializations converge to a poor local
        optimum) is handled at the training level instead, via multiple
        random restarts picked by validation accuracy -- see
        `train_model_with_restarts` in train.py. That is a more honest fix
        than continuing to tune this calibration to chase a single metric.
        """
        self.time_scale = time_activations.abs().mean().clamp_min(1e-6).reshape(1)
        self.freq_scale = freq_activations.abs().mean().clamp_min(1e-6).reshape(1)
        self.calibrated = True

    def forward(self, time_activations: torch.Tensor, freq_activations: torch.Tensor) -> dict:
        if not self.calibrated:
            self.calibrate_scale(time_activations, freq_activations)
        time_scaled = time_activations / self.time_scale
        freq_scaled = freq_activations / self.freq_scale
        combined = torch.cat([time_scaled, freq_scaled], dim=1)
        gate = sparsemax(self.gate_logits(combined))
        gated = combined * gate
        logits = self.classifier(gated)
        return {"logits": logits, "gate": gate, "gated_concepts": gated}


class DualDomainSESM(nn.Module):
    def __init__(self, n_channels: int = 8, n_time_concepts: int = 6, n_freq_concepts: int = 6,
                 n_classes: int = 2, time_kernel_size: int = 30, sample_rate_hz: float = 128.0):
        super().__init__()
        self.time_bank = TimeConceptBank(n_channels, n_time_concepts, time_kernel_size)
        self.freq_bank = FrequencyConceptBank(n_channels, n_freq_concepts, sample_rate_hz=sample_rate_hz)
        self.head = SelectiveHead(n_time_concepts, n_freq_concepts, n_classes)

    def forward(self, x: torch.Tensor) -> dict:
        time_out = self.time_bank(x)
        freq_out = self.freq_bank(x)
        head_out = self.head(time_out["activations"], freq_out["activations"])
        return {
            "logits": head_out["logits"],
            "gate": head_out["gate"],
            "gated_concepts": head_out["gated_concepts"],
            "time_activations": time_out["activations"],
            "time_positions": time_out["positions"],
            "freq_activations": freq_out["activations"],
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x)["logits"].argmax(dim=1)
