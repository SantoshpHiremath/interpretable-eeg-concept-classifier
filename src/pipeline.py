"""End-to-end demo: generate data, train the dual-domain SESM (with multiple
random restarts, keeping the best by validation accuracy -- see train.py for
why this project's sparsemax-gated fusion needs that), evaluate, and print
real example explanations for a few test predictions.
"""
from src.generate_signals import generate_dataset, train_val_test_split
from src.model import DualDomainSESM
from src.train import train_model_with_restarts, evaluate
from src.explain import explain_prediction, format_explanation


def main():
    X, y = generate_dataset(n_samples=400, seed=42, mode="dual")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} channels, {X.shape[2]} timesteps")
    print(f"Train/val/test: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}\n")

    print("Training DualDomainSESM (5 restarts, best kept by validation accuracy)...")
    model, history = train_model_with_restarts(
        lambda: DualDomainSESM(n_channels=X.shape[1]),
        X_train, y_train, X_val, y_val, n_restarts=5, n_epochs=200, patience=60, verbose=True,
    )
    print(f"Best validation accuracy across restarts: {history['best_val_accuracy']:.3f}")

    metrics = evaluate(model, X_test, y_test)
    print(f"\nTest accuracy: {metrics['accuracy']:.3f}")
    print(f"Average concept sparsity (fraction gated to zero): {metrics['avg_sparsity']:.3f}\n")

    print("=" * 70)
    print("EXAMPLE EXPLANATIONS (first 3 test samples)")
    print("=" * 70)
    for i in range(3):
        x = X_test[i:i+1]
        true_label = int(y_test[i].item())
        explanation = explain_prediction(model, x)
        print(f"\nSample {i} (true class: {true_label})")
        print(format_explanation(explanation))


if __name__ == "__main__":
    main()
