"""Single-domain ablation models: architecturally identical to DualDomainSESM
but with one branch's activations replaced by a constant zero tensor BEFORE
the fusion head, for both training and evaluation. This is a stricter and
fairer ablation than simply giving one branch fewer concepts (which still
leaves it a sliver of real capacity) -- a truly disabled branch can
contribute nothing, ever, to the gate or classifier.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model import DualDomainSESM


class TimeOnlyAblation(DualDomainSESM):
    """The frequency branch's activations are always zero -- the model can
    only ever use time-domain concepts to classify.
    """

    def forward(self, x: torch.Tensor) -> dict:
        time_out = self.time_bank(x)
        freq_activations = torch.zeros(x.shape[0], self.freq_bank.n_concepts, device=x.device)
        head_out = self.head(time_out["activations"], freq_activations)
        return {
            "logits": head_out["logits"], "gate": head_out["gate"],
            "gated_concepts": head_out["gated_concepts"],
            "time_activations": time_out["activations"], "time_positions": time_out["positions"],
            "freq_activations": freq_activations,
        }


class FrequencyOnlyAblation(DualDomainSESM):
    """The time branch's activations are always zero -- the model can only
    ever use frequency-domain concepts to classify.
    """

    def forward(self, x: torch.Tensor) -> dict:
        freq_out = self.freq_bank(x)
        time_activations = torch.zeros(x.shape[0], self.time_bank.n_concepts, device=x.device)
        time_positions = torch.zeros(x.shape[0], self.time_bank.n_concepts, dtype=torch.long, device=x.device)
        head_out = self.head(time_activations, freq_out["activations"])
        return {
            "logits": head_out["logits"], "gate": head_out["gate"],
            "gated_concepts": head_out["gated_concepts"],
            "time_activations": time_activations, "time_positions": time_positions,
            "freq_activations": freq_out["activations"],
        }
