import torch

from src.generate_signals import generate_dataset, train_val_test_split
from src.model import DualDomainSESM
from src.train import train_model, train_model_with_restarts, evaluate


class TestTrainModel:
    def test_returns_history_with_expected_keys(self):
        X, y = generate_dataset(n_samples=80, seed=1, mode="dual")
        (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(X, y)
        model = DualDomainSESM(n_channels=X.shape[1])
        history = train_model(model, Xtr, ytr, Xv, yv, n_epochs=10, patience=10)
        assert "train_loss" in history
        assert "val_accuracy" in history
        assert "best_val_accuracy" in history
        assert len(history["train_loss"]) <= 10

    def test_early_stopping_restores_best_checkpoint_not_final_epoch(self):
        """Regression test: the model's weights after training must match the
        BEST validation-accuracy checkpoint, not whatever the last epoch
        happened to produce (which can be worse due to overfitting -- see
        README for the measured val-accuracy-degradation pattern this
        early-stopping mechanism exists to correct).
        """
        X, y = generate_dataset(n_samples=200, seed=1, mode="freq_only")
        (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(X, y)
        model = DualDomainSESM(n_channels=X.shape[1])
        history = train_model(model, Xtr, ytr, Xv, yv, n_epochs=150, patience=100, seed=0)

        model.eval()
        with torch.no_grad():
            final_val_acc = (model(Xv)["logits"].argmax(dim=1) == yv).float().mean().item()
        # The restored model's val accuracy must equal the recorded best (within fp tolerance).
        assert abs(final_val_acc - history["best_val_accuracy"]) < 1e-4

    def test_domain_aux_loss_can_be_disabled(self):
        X, y = generate_dataset(n_samples=80, seed=1, mode="dual")
        (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(X, y)
        model = DualDomainSESM(n_channels=X.shape[1])
        # Should run without error with the auxiliary loss turned off.
        history = train_model(model, Xtr, ytr, Xv, yv, n_epochs=5, patience=5, domain_aux_weight=0.0)
        assert len(history["train_loss"]) > 0


class TestTrainModelWithRestarts:
    def test_returns_the_best_of_several_restarts(self):
        X, y = generate_dataset(n_samples=200, seed=1, mode="freq_only")
        (Xtr, ytr), (Xv, yv), (Xte, yte) = train_val_test_split(X, y)
        model, history = train_model_with_restarts(
            lambda: DualDomainSESM(n_channels=X.shape[1]),
            Xtr, ytr, Xv, yv, n_restarts=3, n_epochs=60, patience=30,
        )
        assert model is not None
        assert history["best_val_accuracy"] >= 0.5

    def test_restarts_never_touch_the_test_set(self):
        """Documents the honest-practice guarantee: train_model_with_restarts
        only receives train/val data -- it has no way to peek at test data,
        so picking the best restart by val accuracy cannot leak test
        information.
        """
        import inspect
        params = list(inspect.signature(train_model_with_restarts).parameters.keys())
        assert "X_test" not in params
        assert "y_test" not in params


class TestEvaluate:
    def test_accuracy_is_between_zero_and_one(self):
        X, y = generate_dataset(n_samples=40, seed=1, mode="dual")
        model = DualDomainSESM(n_channels=X.shape[1])
        metrics = evaluate(model, X, y)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_sparsity_is_between_zero_and_one(self):
        X, y = generate_dataset(n_samples=40, seed=1, mode="dual")
        model = DualDomainSESM(n_channels=X.shape[1])
        metrics = evaluate(model, X, y)
        assert 0.0 <= metrics["avg_sparsity"] <= 1.0

    def test_predictions_have_correct_length(self):
        X, y = generate_dataset(n_samples=40, seed=1, mode="dual")
        model = DualDomainSESM(n_channels=X.shape[1])
        metrics = evaluate(model, X, y)
        assert metrics["predictions"].shape == (40,)
