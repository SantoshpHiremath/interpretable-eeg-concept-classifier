import numpy as np
import torch

from src.generate_signals import (
    generate_dataset, train_val_test_split, N_CHANNELS, N_TIMESTEPS, _waveform_template,
)


class TestGenerateDataset:
    def test_shapes(self):
        X, y = generate_dataset(n_samples=100, seed=1, mode="dual")
        assert X.shape == (100, N_CHANNELS, N_TIMESTEPS)
        assert y.shape == (100,)

    def test_class_balance(self):
        X, y = generate_dataset(n_samples=200, seed=1, mode="dual")
        counts = np.bincount(y.numpy())
        assert counts[0] == counts[1] == 100

    def test_deterministic_given_same_seed(self):
        X1, y1 = generate_dataset(n_samples=50, seed=7, mode="dual")
        X2, y2 = generate_dataset(n_samples=50, seed=7, mode="dual")
        assert torch.equal(X1, X2)
        assert torch.equal(y1, y2)

    def test_different_seeds_produce_different_data(self):
        X1, _ = generate_dataset(n_samples=50, seed=1, mode="dual")
        X2, _ = generate_dataset(n_samples=50, seed=2, mode="dual")
        assert not torch.equal(X1, X2)

    def test_modes_produce_different_data(self):
        X_dual, _ = generate_dataset(n_samples=50, seed=1, mode="dual")
        X_time, _ = generate_dataset(n_samples=50, seed=1, mode="time_only")
        X_freq, _ = generate_dataset(n_samples=50, seed=1, mode="freq_only")
        assert not torch.equal(X_dual, X_time)
        assert not torch.equal(X_dual, X_freq)


class TestTrainValTestSplitIsStratified:
    def test_class_balance_preserved_in_every_split(self):
        """Regression test: a plain sequential slice of a randomly-permuted
        dataset is NOT guaranteed to be class-balanced in the smaller val/
        test slices, and this caused genuinely misleading accuracy readings
        during development (see README). Every split must be class-balanced.
        """
        X, y = generate_dataset(n_samples=400, seed=42, mode="dual")
        (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(X, y)

        for split_y in (ytr, yv, yte):
            counts = np.bincount(split_y.numpy())
            assert counts[0] == counts[1]

    def test_split_sizes_match_requested_fractions(self):
        X, y = generate_dataset(n_samples=400, seed=42, mode="dual")
        (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(X, y, train_frac=0.6, val_frac=0.2)
        assert Xtr.shape[0] == 240
        assert Xv.shape[0] == 80
        assert Xte.shape[0] == 80

    def test_no_overlap_between_splits(self):
        X, y = generate_dataset(n_samples=200, seed=42, mode="dual")
        (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(X, y)
        # Use a per-sample sum as a cheap fingerprint to detect any duplicate row across splits.
        train_fp = set(Xtr.sum(dim=(1, 2)).tolist())
        val_fp = set(Xv.sum(dim=(1, 2)).tolist())
        test_fp = set(Xte.sum(dim=(1, 2)).tolist())
        assert train_fp.isdisjoint(val_fp)
        assert train_fp.isdisjoint(test_fp)
        assert val_fp.isdisjoint(test_fp)


class TestWaveformTemplate:
    def test_template_is_deterministic(self):
        t1 = _waveform_template(30)
        t2 = _waveform_template(30)
        np.testing.assert_array_equal(t1, t2)

    def test_template_length_matches_request(self):
        assert len(_waveform_template(20)) == 20
        assert len(_waveform_template(50)) == 50


class TestOracleTaskDifficulty:
    def test_time_only_task_is_learnable_but_not_trivial(self):
        """A matched filter using the TRUE template must beat chance by a
        wide margin (proving the injected pattern is real and detectable)
        but should not be a perfect/trivial separator (proving the noise
        level is realistic, not decorative) -- this bounds the honest
        difficulty of the time-domain task, referenced throughout the
        README's discussion of the time branch's real, measured ceiling.
        """
        X, y = generate_dataset(n_samples=200, seed=42, mode="time_only")
        template = _waveform_template(30)
        X_np, y_np = X.numpy(), y.numpy()

        correct = 0
        for i in range(len(X_np)):
            sig = X_np[i].mean(axis=0)
            best_pos, best_corr = 0, -2.0
            for pos in range(len(sig) - 30):
                c = np.corrcoef(sig[pos:pos + 30], template)[0, 1]
                if c > best_corr:
                    best_corr, best_pos = c, pos
            pred = 0 if best_pos < len(sig) // 2 else 1
            correct += int(pred == y_np[i])

        oracle_acc = correct / len(X_np)
        assert oracle_acc > 0.75   # clearly better than chance
        assert oracle_acc < 0.98   # not a trivially perfect task
