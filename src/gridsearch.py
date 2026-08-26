"""Local sequential architecture/hyperparameter sweep -- the shape of the
systematic architecture evaluation the posting describes running on a SLURM
cluster, implemented here as a sequential loop (no cluster available in this
environment; see README).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.generate_signals import generate_dataset, train_val_test_split
from src.model import DualDomainSESM
from src.train import train_model, evaluate


@dataclass
class SweepCandidate:
    n_time_concepts: int
    n_freq_concepts: int
    time_kernel_size: int
    lr: float


@dataclass
class SweepResult:
    candidate: SweepCandidate
    val_accuracy: float
    test_accuracy: float


DEFAULT_CANDIDATES = [
    SweepCandidate(n_time_concepts=4, n_freq_concepts=4, time_kernel_size=20, lr=1e-3),
    SweepCandidate(n_time_concepts=6, n_freq_concepts=6, time_kernel_size=30, lr=1e-3),
    SweepCandidate(n_time_concepts=8, n_freq_concepts=8, time_kernel_size=40, lr=5e-4),
]


def run_sweep(candidates: list = None, seed: int = 42, n_epochs: int = 40) -> list:
    if candidates is None:
        candidates = DEFAULT_CANDIDATES
    if not candidates:
        return []

    X, y = generate_dataset(n_samples=300, seed=seed, mode="dual")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train_val_test_split(X, y)

    results = []
    for candidate in candidates:
        model = DualDomainSESM(
            n_channels=X.shape[1], n_time_concepts=candidate.n_time_concepts,
            n_freq_concepts=candidate.n_freq_concepts, time_kernel_size=candidate.time_kernel_size,
        )
        train_model(model, X_train, y_train, X_val, y_val, n_epochs=n_epochs, lr=candidate.lr, seed=seed)
        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)
        results.append(SweepResult(candidate=candidate, val_accuracy=val_metrics["accuracy"],
                                    test_accuracy=test_metrics["accuracy"]))

    return results


def best_by_val_accuracy(results: list):
    if not results:
        return None
    return max(results, key=lambda r: r.val_accuracy)


def format_results_table(results: list) -> str:
    header = f"{'time_c':>6} {'freq_c':>6} {'kernel':>6} {'lr':>8} {'val_acc':>8} {'test_acc':>9}"
    lines = [header, "-" * len(header)]
    for r in sorted(results, key=lambda r: r.val_accuracy, reverse=True):
        c = r.candidate
        lines.append(f"{c.n_time_concepts:>6} {c.n_freq_concepts:>6} {c.time_kernel_size:>6} "
                      f"{c.lr:>8.4f} {r.val_accuracy:>8.3f} {r.test_accuracy:>9.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running local architecture/hyperparameter sweep (3 candidates)...\n")
    results = run_sweep()
    print(format_results_table(results))
    best = best_by_val_accuracy(results)
    print(f"\nBest by validation accuracy: {best.candidate}")
