# Interpretable Time-Frequency Concept Classifier for EEG-Shaped Signals

A tested PyTorch project on interpretable, concept-based deep learning for multi-channel signals. I built a **Self-Explaining Selective Model (SESM)-style** classifier that learns compact, class-specific *concepts* directly from raw signals. The concepts double as the explanation, showing which time- and frequency-domain features drove a classification, and they are combined in a **dual time-frequency architecture**.

## What it does

A black-box classifier explained after the fact (for example with saliency maps or SHAP) is the easier thing to build. `src/model.py`'s `DualDomainSESM` is instead self-explaining by construction: the prediction is a **sparse linear combination of named, inspectable concept activations**. There is no computation path from input to prediction that bypasses the concepts, and this is checked structurally (`test_prediction_head_has_no_direct_input_connection`) rather than only documented.

- **Time-domain concept bank** (`src/concepts.py`, `TimeConceptBank`): a small number of learned 1D-convolutional concept prototypes, one filter per concept, **shared across channels** (see design note 3 below). Each concept's activation is a soft (log-sum-exp) pooled cross-correlation with the input, and the position of its hard-max match is reported separately for localization. The explanation points at real timesteps in the original signal, not an abstract embedding dimension.
- **Frequency-domain concept bank** (`src/concepts.py`, `FrequencyConceptBank`): power-spectral-density features (via `torch.fft.rfft`, banded into 8 frequency bins spanning 0-64 Hz), with learned per-class attention weights over the bins, so each concept activation is traceable to a specific frequency band.
- **Selective fusion and classification head** (`src/model.py`, `SelectiveHead`): a **sparsemax** projection (Martins & Astudillo, 2016, the Euclidean projection onto the probability simplex) over the combined, rescaled time and frequency concept activations, so most concepts are exactly zero for any given prediction. This is the "selective" part of SESM.
- **`src/explain.py`**: turns a prediction into a human-readable explanation, listing the top active time-concepts (with their matched timestep range and channel) and top active frequency-concepts (with their matched band), ranked by contribution weight.
- **`src/gridsearch.py`**: a systematic architecture and hyperparameter sweep, run as a local sequential loop with a results table.

## Data

The data is synthetic. `src/generate_signals.py` generates a multi-channel signal dataset (8 channels, 256 timesteps, sampled at a nominal 128 Hz, the same shape class as real EEG recordings) with **two classes that differ in both a time-domain pattern and a frequency-domain pattern**. This is deliberate: a model that only looked at one domain could not reach high accuracy on the combined task. The signals are generated so I can construct them, break them, and check the results myself, and the pipeline is written so a real EEG dataset can replace the generator. The model is my own concept-based, dual-domain, selective-head design that satisfies the self-explaining property; it is not a reproduction of a specific published architecture.

## Results

- **Single-domain ceiling.** `tests/test_ablation.py` builds a dataset where half the samples are separable only by the time-domain template and half only by the frequency-domain oscillation. A model with access to only one domain is capped near ~0.75 (perfect on its own half, chance on the other), which is checked directly (`test_single_domain_ablations_are_capped_near_the_theoretical_ceiling`).
- **Dual-domain benefit.** The dual-domain model beats the *average* of two architecturally-disabled single-domain ablations that are each trained under the same conditions (`test_dual_domain_model_beats_the_average_single_domain_ablation`). The margin is real and modest, and varies somewhat from run to run.
- **Time-branch ceiling.** The time-domain concept mechanism is a harder optimization than its frequency-domain counterpart (design note 5). An isolated time-only concept bank and classifier reaches roughly a 55-65% ceiling on time-only data, while an oracle matched filter using the true template reaches ~90% on the same data, so the task itself is learnable and the time-concept design leaves room to improve.

## Tests

64 tests (`pytest tests/ -v`), including:

- Sparsemax properties: sums to 1, produces exact zeros (not just small values), concentrates on a dominant logit, and gives every input logit a real, non-NaN gradient (regression test for note 1).
- Concept localization: a hand-built signal with a known injected pattern at a known position is localized by a concept bank whose prototype is that pattern (within ±2 timesteps); a pure sine wave at a known frequency has its power identified in the containing band.
- Architectural self-explaining check: the classification head's `forward` signature only accepts concept activations, never a raw signal, and no head parameter has a dimension matching the raw timestep count.
- Fixed-scale calibration: set once from the first batch and unchanged by a very different second batch (regression test distinguishing this from BatchNorm/LayerNorm, note 2).
- Parameter sharing: the time-concept bank's parameter count has no `n_channels` factor (regression test for note 3).
- Stratified split: every split (train/val/test) is class-balanced, and there is no sample overlap between splits (regression test for note 4).
- Early stopping: the model's final weights match the best recorded validation checkpoint, not whatever the last epoch happened to produce.
- Ablation tests: single-domain data is solved reliably when only that domain is informative; the dual model beats the average single-domain ablation on a mixed-cue task; a disabled domain's gated contribution to the classifier is always exactly zero.
- Explanation faithfulness: every contribution in a generated explanation corresponds to a concept with strictly positive gate weight (never a zeroed-out concept), and a time-concept's reported timestep range is checked against the model's own recorded position tensor.

## Project structure

```
src/
  generate_signals.py   synthetic multi-channel signal generator and stratified split
  mixed_dataset.py      mixed-cue dataset for the ablation task
  concepts.py           TimeConceptBank and FrequencyConceptBank
  model.py              DualDomainSESM, SelectiveHead, sparsemax
  train.py              training, early stopping, multi-restart selection
  ablation.py           single-domain ablations
  explain.py            human-readable explanations from a prediction
  gridsearch.py         local architecture/hyperparameter sweep
  pipeline.py           end-to-end train, evaluate, explain
tests/                  64 tests
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python3 -m src.pipeline          # trains (5 restarts, best kept), evaluates, prints explanations
python3 -m src.gridsearch        # small local architecture/hyperparameter sweep with a results table
pytest tests/ -v                  # 64 tests (the ablation suite takes ~1-2 minutes; it trains several models)
```

## Notes

Getting a first version training was fast. Getting it to train *correctly*, using both domains, generalizing rather than memorizing, and being fairly evaluated, came down to five design decisions, each found, diagnosed, and fixed with a regression test.

**1. Sparsemax gate instead of a hard mask.** The first version of the selective gate used softmax followed by a hard mean-threshold mask. A boolean mask has no gradient, so every masked-out logit received zero gradient and training collapsed to predicting from one arbitrary concept. Real sparsemax is piecewise-linear and differentiable almost everywhere, so every logit gets a gradient even on steps where its output is exactly zero (`test_gradient_flows_to_all_logits_not_just_selected_ones`).

**2. Fixed one-time rescale of the two domains.** Frequency activations were ~70x larger than time activations, because power-spectral-density features and correlation-based features start on different numeric scales. BatchNorm and LayerNorm were both tried and both made things worse: forcing every batch or sample to unit variance inflates a domain's pure noise up to the same apparent scale as the other domain's real signal (measured on data where only the time domain carried signal, where validation accuracy stayed at chance). The fix is a **fixed, one-time rescale** calibrated from a single batch at the start of training (`SelectiveHead.calibrate_scale`), which corrects for units without re-equalizing informativeness on every forward pass.

**3. Time-concept filters shared across channels.** The first `TimeConceptBank` used one independent `Conv1d` filter per channel (`n_concepts × n_channels × kernel_size` parameters). The data-generating process injects the *same* template shape on every channel plus independent per-channel noise, so per-channel filters overfit to the noise. The model reached 100% train accuracy but stayed at chance-level validation accuracy across 4 seeds, even though an oracle matched filter achieves ~90% on the same held-out data. Tying one filter across all channels via a reshape-and-average cut this bank's parameter count 8x (`test_filter_is_shared_across_channels_not_independent_per_channel`).

**4. Class-stratified train/val/test split.** A random permutation followed by a sequential slice does not guarantee balanced classes in the smaller val/test slices. One split landed at 54/26 (67%/33%) in validation purely by chance, so an untrained model's majority-class baseline (0.675) looked like a real partial signal. A per-class stratified split fixed this (`test_class_balance_preserved_in_every_split`).

**5. Multiple restarts for a seed-sensitive optimization.** The sparsemax-gated fusion is a non-convex optimization: with identical data, changing only the random seed swung validation accuracy from chance to 100%. This is a property of the architecture (a gate can hard-select one domain early in training and struggle to leave that choice). I handle it with multiple random restarts, kept by validation accuracy only (`train_model_with_restarts` in `train.py`); the test set is never touched during selection. An auxiliary loss that separately supervises a time-only and a frequency-only sub-prediction during training (`domain_aux_weight` in `train_model`) further encourages both branches to become independently useful.

Training runs on CPU, and the grid search runs as a sequential local loop rather than a cluster-scheduled job array.

## Possible extensions

- Run the pipeline on a public EEG dataset (for example PhysioNet or the BCI Competition data).
- Improve the time-concept mechanism toward the oracle matched-filter ceiling.
- Schedule the grid search as a job array on a multi-node cluster.
