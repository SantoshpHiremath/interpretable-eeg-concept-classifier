import numpy as np
import torch

from src.concepts import TimeConceptBank, FrequencyConceptBank
from src.generate_signals import _waveform_template, N_TIMESTEPS


class TestTimeConceptBankShapes:
    def test_output_shapes(self):
        bank = TimeConceptBank(n_channels=8, n_concepts=6, kernel_size=30)
        x = torch.randn(5, 8, N_TIMESTEPS)
        out = bank(x)
        assert out["activations"].shape == (5, 6)
        assert out["positions"].shape == (5, 6)

    def test_activations_are_non_negative(self):
        bank = TimeConceptBank(n_channels=8, n_concepts=6, kernel_size=30)
        x = torch.randn(10, 8, N_TIMESTEPS)
        out = bank(x)
        assert (out["activations"] >= 0).all()

    def test_positions_are_valid_timestep_indices(self):
        bank = TimeConceptBank(n_channels=8, n_concepts=6, kernel_size=30)
        x = torch.randn(5, 8, N_TIMESTEPS)
        out = bank(x)
        max_valid = N_TIMESTEPS - 30
        assert (out["positions"] >= 0).all()
        assert (out["positions"] <= max_valid).all()


class TestTimeConceptLocalization:
    def test_localizes_to_the_actual_injected_pattern_position(self):
        """A hand-built signal with a known, exact template at a known
        position: a concept bank whose prototype IS that template should
        report a position at (or very near) the true injection point.
        """
        template = _waveform_template(30)
        signal = np.zeros(N_TIMESTEPS, dtype=np.float32)
        true_start = 100
        signal[true_start:true_start + 30] += template.astype(np.float32) * 2.0
        x = torch.from_numpy(signal).reshape(1, 1, N_TIMESTEPS).repeat(1, 4, 1)

        bank = TimeConceptBank(n_channels=4, n_concepts=1, kernel_size=30)
        with torch.no_grad():
            bank.prototypes.weight.copy_(torch.from_numpy(template).reshape(1, 1, 30).float())

        out = bank(x)
        reported_start = int(out["positions"][0, 0].item())
        assert abs(reported_start - true_start) <= 2  # allow a small off-by-few from discretization

    def test_matched_filter_gives_high_activation_for_matching_signal(self):
        template = _waveform_template(30)
        signal = np.zeros(N_TIMESTEPS, dtype=np.float32)
        signal[50:80] += template.astype(np.float32) * 2.0
        x_match = torch.from_numpy(signal).reshape(1, 1, N_TIMESTEPS).repeat(1, 4, 1)
        x_noise = torch.randn(1, 4, N_TIMESTEPS) * 0.1

        bank = TimeConceptBank(n_channels=4, n_concepts=1, kernel_size=30)
        with torch.no_grad():
            bank.prototypes.weight.copy_(torch.from_numpy(template).reshape(1, 1, 30).float())

        act_match = bank(x_match)["activations"]
        act_noise = bank(x_noise)["activations"]
        assert act_match.item() > act_noise.item()


class TestTimeConceptBankParameterSharing:
    def test_filter_is_shared_across_channels_not_independent_per_channel(self):
        """Regression test for the channel-overparameterization bug (see
        README): the prototype conv must have a SINGLE input channel, applied
        identically to every channel of the signal, not one independent
        filter per channel.
        """
        bank = TimeConceptBank(n_channels=8, n_concepts=6, kernel_size=30)
        assert bank.prototypes.in_channels == 1
        # Parameter count must be n_concepts * kernel_size (no n_channels factor).
        assert bank.prototypes.weight.numel() == 6 * 30


class TestFrequencyConceptBank:
    def test_output_shapes(self):
        bank = FrequencyConceptBank(n_channels=8, n_concepts=6)
        x = torch.randn(5, 8, N_TIMESTEPS)
        out = bank(x)
        assert out["activations"].shape == (5, 6)

    def test_psd_shape(self):
        bank = FrequencyConceptBank(n_channels=8, n_concepts=6, n_bands=8)
        x = torch.randn(5, 8, N_TIMESTEPS)
        psd = bank.power_spectral_density(x)
        assert psd.shape == (5, 8, 8)

    def test_psd_is_non_negative(self):
        """Power is |FFT|^2 -- must never be negative."""
        bank = FrequencyConceptBank(n_channels=8, n_concepts=6)
        x = torch.randn(5, 8, N_TIMESTEPS)
        psd = bank.power_spectral_density(x)
        assert (psd >= 0).all()

    def test_dominant_band_correctly_identifies_injected_oscillation(self):
        """A pure sine wave at a known frequency should have its power
        concentrated in the band containing that frequency.
        """
        sample_rate = 128.0
        t = torch.arange(N_TIMESTEPS) / sample_rate
        dominant_hz = 20.0
        signal = torch.sin(2 * np.pi * dominant_hz * t)
        x = signal.reshape(1, 1, N_TIMESTEPS).repeat(1, 4, 1)

        bank = FrequencyConceptBank(n_channels=4, n_concepts=1, n_bands=8, sample_rate_hz=sample_rate)
        psd = bank.power_spectral_density(x)
        band_powers = psd[0].mean(dim=0)  # average across channels
        peak_band = int(band_powers.argmax().item())

        edges = bank.band_edges_hz(N_TIMESTEPS)
        lo, hi = edges[peak_band].item(), edges[peak_band + 1].item()
        assert lo <= dominant_hz <= hi

    def test_dominant_band_index_matches_highest_attention_weight(self):
        bank = FrequencyConceptBank(n_channels=4, n_concepts=3, n_bands=8)
        with torch.no_grad():
            bank.band_attention.zero_()
            bank.band_attention[0, :, 5] = 10.0  # force concept 0's peak band to index 5
        assert bank.dominant_band_index(0) == 5
