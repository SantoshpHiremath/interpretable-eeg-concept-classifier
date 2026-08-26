# Interpretable Time-Frequency Concept Classifier for EEG-Shaped Signals

A real, tested PyTorch project built specifically to close a gap for
XITASO's "Werkstudent — Interpretierbare Machine-Learning-Methoden für
EEG-basierte Klassifizierung" posting (EXACT-EEG project): the posting's
core ask is a **Self-Explaining Selective Model (SESM)** that learns
compact, class-specific *concepts* directly from raw signals, where
those concepts double as the explanation — showing which time- and
frequency-domain features drove a classification — combined in a
**dual time-frequency architecture**. Nothing in my prior project
portfolio touched interpretable/concept-based deep learning or
biomedical signals at all, so this is built from scratch, in that exact
shape, rather than adjacent evidence repackaged.

## What this is (read before citing anywhere)

**There is no real EEG data, and no clinical claim here.** I have no
access to a real EEG dataset (PhysioNet, TUH EEG Corpus, BCI Competition
data, and similar all require registration/download access unavailable
in this environment) and no neuroscience or clinical background.
`src/generate_signals.py` generates a synthetic multi-channel signal
dataset (8 channels, 256 timesteps, sampled at a nominal 128 Hz — the
same shape class as real EEG recordings) with **two classes that differ
in BOTH a time-domain pattern and a frequency-domain pattern**,
deliberately, so a model that only looked at one domain could not
reach high accuracy on the combined task. This is not a SLURM cluster
either — there's no multi-node GPU scheduler available here, so
`src/gridsearch.py` runs the same kind of systematic architecture/
hyperparameter sweep the posting names, as a local sequential loop with
a results table, rather than a cluster-scheduled job array.

If asked in an interview: I haven't worked with real EEG data, SLURM, or
a production self-explaining architecture before this project. What's
built here is a genuine, working implementation of the *mechanism* the
posting describes — concept-based, dual-domain, self-explaining
classification — built, debugged, and honestly evaluated on synthetic
signals I could construct, break, and check myself.

## The architecture (what "self-explaining" actually means here)

A black-box classifier explained after the fact (e.g. saliency maps,
SHAP) is the easier thing to build, and it's explicitly not what the
posting asks for. `src/model.py`'s `DualDomainSESM` is self-explaining
by construction: the prediction is a **sparse linear combination of
named, inspectable concept activations** — there is no computation path
from input to prediction that bypasses the concepts, which is checked
structurally (`test_prediction_head_has_no_direct_input_connection`),
not just documented.

- **Time-domain concept bank** (`src/concepts.py`, `TimeConceptBank`) —
  a small number of learned 1D-convolutional concept prototypes, one
  filter per concept, **shared across channels** (not one independent
  filter per channel — see the parameter-sharing bug below for why that
  matters). Each concept's activation is a soft (log-sum-exp) pooled
  cross-correlation with the input, and the position of its hard-max
  match is reported separately for localization — an explanation that
  points at real timesteps in the original signal, not an abstract
  embedding dimension.
- **Frequency-domain concept bank** (`src/concepts.py`,
  `FrequencyConceptBank`) — real power-spectral-density features (via
  `torch.fft.rfft`, banded into 8 frequency bins spanning 0–64 Hz),
  with learned per-class attention weights over the bins, so each
  concept activation is traceable to a specific, real frequency band.
- **Selective fusion + classification head** (`src/model.py`,
  `SelectiveHead`) — a real **sparsemax** projection (Martins &
  Astudillo, 2016 — the Euclidean projection onto the probability
  simplex) over the combined, rescaled time+frequency concept
  activations, so most concepts are exactly zero for any given
  prediction — the "selective" part of SESM.
- **`src/explain.py`** — turns a prediction into a human-readable
  explanation: the top active time-concepts (with their matched
  timestep range and channel) and top active frequency-concepts (with
  their matched band), ranked by contribution weight.

## Debugging this was most of the actual work — five real, distinct bugs

Getting a first version training was fast; getting it to train
*correctly* — actually using both domains, generalizing rather than
memorizing, and being fairly evaluated — surfaced five separate, real
bugs, each independently found, diagnosed, and fixed. This section is
long on purpose: it's the most honest evidence in this project of an
actual debugging process, not a polished-after-the-fact description.

**1. The gate collapsed to a single always-on concept.** The first
version of the "selective" sparse gate used softmax followed by a hard
mean-threshold mask. A boolean mask has no gradient, so every masked-out
logit received zero gradient every step — training collapsed to
predicting from one arbitrary concept regardless of input. Fixed by
implementing real sparsemax, whose projection is piecewise-linear and
differentiable almost everywhere, so every logit gets a real gradient
even on steps where its output is exactly zero
(`test_gradient_flows_to_all_logits_not_just_selected_ones`).

**2. Frequency activations were ~70x larger than time activations.**
Power-spectral-density-derived features and correlation-based features
start on completely different numeric scales. Unnormalized, the gate and
classifier were dominated by whichever domain had larger raw magnitude,
not whichever was more predictive. BatchNorm and LayerNorm were both
tried as fixes and both made things *worse*: forcing every batch/sample
to unit variance inflates a domain's pure noise (when that domain isn't
informative for a given dataset) up to the same apparent scale as the
other domain's real signal — measured directly on data where only the
time domain carried signal, val accuracy stayed at chance regardless of
patience. The actual fix is a **fixed, one-time rescale** calibrated
from a single batch at the start of training (`SelectiveHead.
calibrate_scale`), which corrects for units without re-equalizing
informativeness on every forward pass.

**3. Time-concept filters had 8x more parameters than the task's real
structure needs.** The first `TimeConceptBank` used one independent
`Conv1d` filter per channel (`n_concepts × n_channels × kernel_size`
parameters). The data-generating process injects the *same* template
shape on every channel plus independent per-channel noise — so a
per-channel-independent filter systematically overfits to per-channel
noise instead of the shared template. Confirmed directly: the model
reached 100% train accuracy but stayed at chance-level validation
accuracy across 4 seeds, even though an oracle matched filter using the
*true* template achieves ~90% on the same held-out data (proving the
task itself is learnable). Fixed by tying one filter across all
channels via a reshape-and-average, cutting this bank's parameter count
8x (`test_filter_is_shared_across_channels_not_independent_per_channel`).

**4. The train/val/test split wasn't class-stratified.** A random
permutation followed by a sequential slice does *not* guarantee balanced
classes in the smaller val/test slices — one real split landed at
54/26 (67%/33%) in validation purely by chance. This produced
genuinely misleading debugging signal for an embarrassingly long time:
an untrained model's val accuracy (the majority-class baseline, 0.675)
looked like a real partial signal, and a model that was actually
learning real structure looked like it was "getting worse during
training" because it was moving *away* from that deceptively strong
majority-class baseline. Fixed with a proper per-class stratified split
(`test_class_balance_preserved_in_every_split`).

**5. The sparsemax-gated fusion is a seed-sensitive, non-convex
optimization.** Even after fixes 1–4, identical data with only the
random seed changed swung from chance-level to 100% validation
accuracy. This is a genuine property of this architecture (a gate that
can hard-select one domain over another early in training and then
struggle to escape that choice), not a remaining bug — the honest fix is
multiple random restarts, kept only by validation accuracy
(`train_model_with_restarts` in `train.py`), which is a standard
technique for exactly this kind of landscape and is disclosed here
rather than silently reporting only a best-case single run. An
auxiliary loss that separately supervises a time-only and a
frequency-only sub-prediction during training (`domain_aux_weight` in
`train_model`) further encourages both branches to become independently
useful rather than the easier one dominating.

## Proving the dual-domain architecture does real work — and an honest limit

`tests/test_ablation.py` builds a dataset where half the samples are
separable only by the time-domain template, half only by the
frequency-domain oscillation — a model with access to only one domain is
mathematically capped near ~0.75 (perfect on its own half, chance on the
other), which is checked directly
(`test_single_domain_ablations_are_capped_near_the_theoretical_ceiling`).
The dual-domain model beats the *average* of two honestly-trained,
architecturally-disabled single-domain ablations
(`test_dual_domain_model_beats_the_average_single_domain_ablation`).

**This margin is real but modest, and that's disclosed rather than
tuned away.** Repeated runs show the dual model beating single-domain
ablations by a real but sometimes small amount, not a dramatic "dual
solves everything, single-domain solves nothing" result. The root cause
is bug/finding #5 above, restated at the whole-task level: this
project's time-domain concept mechanism is a genuinely harder
optimization than its frequency-domain counterpart, so even when the
gate correctly identifies that a sample needs the time domain, the
time-branch's own prediction on that sample is itself imperfect
(measured directly: an isolated time-only concept bank + classifier
reaches roughly a 55–65% ceiling on time-only data, well below the ~90%
an oracle matched filter achieves on the same data). This is reported as
a real, specific limitation of the current time-concept design, not
smoothed over.

## Verification

64 tests (`pytest tests/ -v`), including:

- Sparsemax properties: sums to 1, produces exact zeros (not just small
  values), concentrates on a dominant logit, and — the regression test
  for bug #1 — gives every input logit a real, non-NaN gradient.
- Concept localization: a hand-built signal with a known injected
  pattern at a known position is correctly localized by a concept bank
  whose prototype IS that pattern (within ±2 timesteps); a pure sine
  wave at a known frequency has its power correctly identified in the
  containing band.
- Architectural self-explaining check: the classification head's
  `forward` signature only accepts concept activations, never a raw
  signal, and no head parameter has a dimension matching the raw
  timestep count.
- Fixed-scale calibration: set once from the first batch, unchanged by
  a wildly different second batch (regression test distinguishing this
  from BatchNorm/LayerNorm, bug #2).
- Parameter-sharing: the time-concept bank's parameter count has no
  `n_channels` factor (regression test for bug #3).
- Stratified split: every split (train/val/test) is class-balanced, and
  there is no sample overlap between splits (regression test for bug
  #4).
- Early stopping: the model's final weights match the best recorded
  validation checkpoint, not whatever the last epoch happened to
  produce.
- Ablation tests (above): single-domain data is solved reliably when
  only that domain is informative; the dual model beats the average
  single-domain ablation on a mixed-cue task; a disabled domain's gated
  contribution to the classifier is always exactly zero.
- Explanation faithfulness: every contribution in a generated
  explanation corresponds to a concept with strictly positive gate
  weight (never a zeroed-out concept); a time-concept's reported
  timestep range is checked against the model's own recorded position
  tensor, not just that some text was generated.

## Running it

```bash
pip install -r requirements.txt
python3 -m src.pipeline          # trains (5 restarts, best kept), evaluates, prints real explanations
python3 -m src.gridsearch        # small local architecture/hyperparameter sweep with a results table
pytest tests/ -v                  # 64 tests (the ablation suite takes ~1-2 minutes; it trains several models)
```

## What this doesn't demonstrate

This project doesn't use real EEG data, doesn't run on a SLURM cluster
or any multi-node scheduler, doesn't implement the specific published
SESM architecture verbatim (this is my own concept-based, dual-domain,
selective-head design built to satisfy the same self-explaining
property, not a reproduction of a paper I haven't read in full), and —
disclosed at length above — its time-domain concept mechanism has a
real, measured accuracy ceiling well below what the frequency-domain
mechanism or an oracle matched filter can achieve on the same data. It
demonstrates real, working, verified concept-based self-explaining
classification across combined time and frequency domains, with concept
faithfulness checked structurally rather than assumed, and with an
honest, detailed account of what broke, why, and how each fix was
verified — built and tested on synthetic signals I could construct and
check myself.
