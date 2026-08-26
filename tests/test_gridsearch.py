from src.gridsearch import (
    run_sweep, best_by_val_accuracy, format_results_table, SweepCandidate, SweepResult,
)


class TestRunSweep:
    def test_returns_one_result_per_candidate(self):
        candidates = [
            SweepCandidate(n_time_concepts=3, n_freq_concepts=3, time_kernel_size=20, lr=1e-3),
            SweepCandidate(n_time_concepts=4, n_freq_concepts=4, time_kernel_size=25, lr=1e-3),
        ]
        results = run_sweep(candidates=candidates, seed=1, n_epochs=10)
        assert len(results) == 2

    def test_empty_candidate_list_does_not_crash(self):
        results = run_sweep(candidates=[], seed=1, n_epochs=10)
        assert results == []

    def test_each_result_has_valid_accuracy_range(self):
        candidates = [SweepCandidate(n_time_concepts=3, n_freq_concepts=3, time_kernel_size=20, lr=1e-3)]
        results = run_sweep(candidates=candidates, seed=1, n_epochs=10)
        for r in results:
            assert 0.0 <= r.val_accuracy <= 1.0
            assert 0.0 <= r.test_accuracy <= 1.0


class TestBestByValAccuracy:
    def test_selects_the_highest_val_accuracy_candidate(self):
        c1 = SweepCandidate(n_time_concepts=3, n_freq_concepts=3, time_kernel_size=20, lr=1e-3)
        c2 = SweepCandidate(n_time_concepts=4, n_freq_concepts=4, time_kernel_size=25, lr=1e-3)
        results = [
            SweepResult(candidate=c1, val_accuracy=0.6, test_accuracy=0.55),
            SweepResult(candidate=c2, val_accuracy=0.9, test_accuracy=0.85),
        ]
        best = best_by_val_accuracy(results)
        assert best.candidate is c2

    def test_empty_results_returns_none(self):
        assert best_by_val_accuracy([]) is None


class TestFormatResultsTable:
    def test_produces_non_empty_string(self):
        c1 = SweepCandidate(n_time_concepts=3, n_freq_concepts=3, time_kernel_size=20, lr=1e-3)
        results = [SweepResult(candidate=c1, val_accuracy=0.7, test_accuracy=0.65)]
        table = format_results_table(results)
        assert len(table) > 0
        assert "0.700" in table

    def test_sorted_descending_by_val_accuracy(self):
        c1 = SweepCandidate(n_time_concepts=3, n_freq_concepts=3, time_kernel_size=20, lr=1e-3)
        c2 = SweepCandidate(n_time_concepts=4, n_freq_concepts=4, time_kernel_size=25, lr=1e-3)
        results = [
            SweepResult(candidate=c1, val_accuracy=0.5, test_accuracy=0.5),
            SweepResult(candidate=c2, val_accuracy=0.9, test_accuracy=0.9),
        ]
        table = format_results_table(results)
        lines = table.strip().split("\n")
        # The row for c2 (0.9) must appear before the row for c1 (0.5).
        idx_09 = next(i for i, l in enumerate(lines) if "0.900" in l)
        idx_05 = next(i for i, l in enumerate(lines) if "0.500" in l)
        assert idx_09 < idx_05
