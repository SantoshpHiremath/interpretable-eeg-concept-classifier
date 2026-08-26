import torch

from src.explain import explain_prediction, format_explanation, Explanation, ConceptContribution
from src.model import DualDomainSESM


class TestExplainPrediction:
    def test_returns_explanation_with_valid_predicted_class(self):
        model = DualDomainSESM(n_channels=8)
        x = torch.randn(1, 8, 256)
        explanation = explain_prediction(model, x)
        assert explanation.predicted_class in (0, 1)

    def test_contributions_are_sorted_by_weight_descending(self):
        model = DualDomainSESM(n_channels=8)
        x = torch.randn(1, 8, 256)
        explanation = explain_prediction(model, x)
        weights = [c.weight for c in explanation.contributions]
        assert weights == sorted(weights, reverse=True)

    def test_only_non_zero_gated_concepts_appear_in_the_explanation(self):
        """The explanation must reflect the SELECTIVE (sparse) gate -- a
        concept with exactly zero gate weight must never appear, since it
        contributed nothing to the prediction.
        """
        model = DualDomainSESM(n_channels=8)
        x = torch.randn(1, 8, 256)
        with torch.no_grad():
            gate = model(x)["gate"][0]
        explanation = explain_prediction(model, x, top_k=20)  # request more than could exist
        n_time = model.time_bank.n_concepts
        for c in explanation.contributions:
            idx = c.concept_index if c.concept_type == "time" else n_time + c.concept_index
            assert gate[idx].item() > 0

    def test_time_contribution_detail_matches_actual_activation_position(self):
        """The reported timestep range must come from the model's own
        recorded position, not be fabricated -- checked against the raw
        tensor, not just that some text was generated.
        """
        model = DualDomainSESM(n_channels=8, time_kernel_size=30)
        x = torch.randn(1, 8, 256)
        with torch.no_grad():
            out = model(x)
        explanation = explain_prediction(model, x, top_k=20)
        time_contribs = [c for c in explanation.contributions if c.concept_type == "time"]
        for c in time_contribs:
            expected_start = int(out["time_positions"][0, c.concept_index].item())
            assert c.detail == f"timesteps {expected_start}-{expected_start + 30}"

    def test_frequency_contribution_detail_is_a_valid_hz_range(self):
        model = DualDomainSESM(n_channels=8, sample_rate_hz=128.0)
        x = torch.randn(1, 8, 256)
        explanation = explain_prediction(model, x, top_k=20)
        freq_contribs = [c for c in explanation.contributions if c.concept_type == "frequency"]
        for c in freq_contribs:
            assert "Hz" in c.detail
            lo_str, rest = c.detail.split("-")
            hi_str = rest.replace(" Hz", "")
            assert 0.0 <= float(lo_str) < float(hi_str) <= 64.0

    def test_top_k_limits_the_number_of_contributions(self):
        model = DualDomainSESM(n_channels=8)
        x = torch.randn(1, 8, 256)
        explanation = explain_prediction(model, x, top_k=2)
        assert len(explanation.contributions) <= 2


class TestFormatExplanation:
    def test_format_includes_predicted_class(self):
        explanation = Explanation(predicted_class=1, contributions=[])
        text = format_explanation(explanation)
        assert "1" in text

    def test_format_includes_each_contribution(self):
        explanation = Explanation(
            predicted_class=0,
            contributions=[
                ConceptContribution(concept_type="time", concept_index=2, weight=0.8, detail="timesteps 10-40"),
                ConceptContribution(concept_type="frequency", concept_index=1, weight=0.2, detail="8.0-16.0 Hz"),
            ],
        )
        text = format_explanation(explanation)
        assert "time concept 2" in text
        assert "timesteps 10-40" in text
        assert "frequency concept 1" in text
        assert "8.0-16.0 Hz" in text
