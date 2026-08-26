"""Turns a single prediction's concept activations into a human-readable
explanation: which time-concepts and frequency-concepts were active, where
they matched in time, and which frequency band they correspond to.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.model import DualDomainSESM


@dataclass
class ConceptContribution:
    concept_type: str  # "time" | "frequency"
    concept_index: int
    weight: float
    detail: str  # e.g. "timesteps 28-58" or "8.0-16.0 Hz"


@dataclass
class Explanation:
    predicted_class: int
    contributions: list  # list[ConceptContribution], sorted by weight descending


def explain_prediction(model: DualDomainSESM, x: torch.Tensor, sample_rate_hz: float = 128.0,
                        top_k: int = 4) -> Explanation:
    """x: a single sample, shape (1, n_channels, n_timesteps)."""
    model.eval()
    with torch.no_grad():
        out = model(x)

    predicted_class = int(out["logits"].argmax(dim=1).item())
    gate = out["gate"][0]  # (n_time_concepts + n_freq_concepts,)
    n_time = model.time_bank.n_concepts
    kernel_size = model.time_bank.kernel_size

    contributions = []
    for i in range(n_time):
        w = gate[i].item()
        if w <= 0:
            continue
        start = int(out["time_positions"][0, i].item())
        end = start + kernel_size
        contributions.append(ConceptContribution(
            concept_type="time", concept_index=i, weight=w,
            detail=f"timesteps {start}-{end}",
        ))

    edges = model.freq_bank.band_edges_hz(x.shape[2])
    for j in range(model.freq_bank.n_concepts):
        gi = n_time + j
        w = gate[gi].item()
        if w <= 0:
            continue
        band_idx = model.freq_bank.dominant_band_index(j)
        lo, hi = edges[band_idx].item(), edges[band_idx + 1].item()
        contributions.append(ConceptContribution(
            concept_type="frequency", concept_index=j, weight=w,
            detail=f"{lo:.1f}-{hi:.1f} Hz",
        ))

    contributions.sort(key=lambda c: c.weight, reverse=True)
    return Explanation(predicted_class=predicted_class, contributions=contributions[:top_k])


def format_explanation(explanation: Explanation) -> str:
    lines = [f"Predicted class: {explanation.predicted_class}"]
    for c in explanation.contributions:
        lines.append(f"  [{c.concept_type} concept {c.concept_index}] weight={c.weight:.3f}  ({c.detail})")
    return "\n".join(lines)
