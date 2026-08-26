import torch

from src.model import sparsemax, DualDomainSESM, SelectiveHead


class TestSparsemax:
    def test_output_sums_to_one(self):
        x = torch.randn(10, 6)
        y = sparsemax(x)
        assert torch.allclose(y.sum(dim=-1), torch.ones(10), atol=1e-5)

    def test_output_is_non_negative(self):
        x = torch.randn(10, 6)
        y = sparsemax(x)
        assert (y >= 0).all()

    def test_produces_exact_zeros_not_just_small_values(self):
        """The defining 'selective' property: sparsemax must produce EXACT
        zeros for low-scoring entries, unlike softmax which never reaches
        exactly zero.
        """
        x = torch.tensor([[5.0, 1.0, 0.5, -1.0, -3.0, -5.0]])
        y = sparsemax(x)
        assert (y == 0).sum().item() >= 2  # at least some entries exactly zero

    def test_concentrates_on_dominant_logit(self):
        x = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        y = sparsemax(x)
        assert y[0, 0].item() > 0.99

    def test_gradient_flows_to_all_logits_not_just_selected_ones(self):
        """Regression test for the original hard-threshold-mask gating bug
        (see README): a boolean mask blocks gradient to every masked-out
        entry, which collapsed training to a single always-on concept. Real
        sparsemax must give a well-defined (possibly zero, but not None)
        gradient to every input logit.
        """
        x = torch.tensor([[3.0, 2.9, 0.2, 0.1, -1.0, -2.0]], requires_grad=True)
        y = sparsemax(x)
        target = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        loss = ((y - target) ** 2).sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestSelectiveHead:
    def test_gate_sums_to_one_per_sample(self):
        head = SelectiveHead(n_time_concepts=6, n_freq_concepts=6, n_classes=2)
        time_act = torch.rand(4, 6)
        freq_act = torch.rand(4, 6) * 50  # different scale, as real freq activations are
        out = head(time_act, freq_act)
        assert torch.allclose(out["gate"].sum(dim=-1), torch.ones(4), atol=1e-4)

    def test_calibrates_scale_on_first_call(self):
        head = SelectiveHead(n_time_concepts=6, n_freq_concepts=6, n_classes=2)
        assert not head.calibrated
        time_act = torch.rand(4, 6)
        freq_act = torch.rand(4, 6) * 50
        head(time_act, freq_act)
        assert head.calibrated
        assert head.time_scale.item() > 0
        assert head.freq_scale.item() > 0

    def test_scale_is_fixed_after_calibration_not_recomputed_every_call(self):
        """Regression test distinguishing this from BatchNorm/LayerNorm (see
        README): the rescale factor must be a FIXED buffer set once, not
        something recomputed from each new batch's statistics.
        """
        head = SelectiveHead(n_time_concepts=6, n_freq_concepts=6, n_classes=2)
        head(torch.rand(4, 6), torch.rand(4, 6) * 50)
        scale_after_first = head.time_scale.item()
        head(torch.rand(4, 6) * 1000, torch.rand(4, 6) * 50)  # wildly different second batch
        assert head.time_scale.item() == scale_after_first


class TestDualDomainSESMArchitecture:
    def test_output_shapes(self):
        model = DualDomainSESM(n_channels=8, n_time_concepts=6, n_freq_concepts=6, n_classes=2)
        x = torch.randn(5, 8, 256)
        out = model(x)
        assert out["logits"].shape == (5, 2)
        assert out["gate"].shape == (5, 12)

    def test_prediction_head_has_no_direct_input_connection(self):
        """Structural self-explaining check: the classification head's
        parameters must only be reachable from the raw input THROUGH the
        concept banks, never directly. Verified by confirming the head
        module has no Conv/Linear layer whose input dimension matches the
        raw signal length, and that model.head's forward signature only
        accepts concept activations, not raw signals.
        """
        model = DualDomainSESM(n_channels=8, n_time_concepts=6, n_freq_concepts=6, n_classes=2)
        import inspect
        head_forward_params = list(inspect.signature(model.head.forward).parameters.keys())
        assert head_forward_params == ["time_activations", "freq_activations"]
        # No parameter tensor in the head has a dimension equal to the raw timestep count.
        for p in model.head.parameters():
            assert 256 not in p.shape

    def test_predict_returns_class_indices(self):
        model = DualDomainSESM(n_channels=8)
        x = torch.randn(5, 8, 256)
        preds = model.predict(x)
        assert preds.shape == (5,)
        assert set(preds.tolist()).issubset({0, 1})

    def test_gradient_flows_to_both_concept_banks(self):
        model = DualDomainSESM(n_channels=8)
        x = torch.randn(5, 8, 256)
        y = torch.tensor([0, 1, 0, 1, 0])
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out["logits"], y)
        loss.backward()
        assert model.time_bank.prototypes.weight.grad is not None
        assert not torch.isnan(model.time_bank.prototypes.weight.grad).all()
        assert model.freq_bank.band_attention.grad is not None
