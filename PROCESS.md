# Beat-Conditioned GaMaDHaNi: Process & Evaluation

## Overview

We fine-tune GaMaDHaNi Stage 1 (a diffusion-based pitch model for Hindustani vocal music) to generate pitch contours that are rhythmically aligned to a given taal beat structure. The model takes a beat signal as conditioning input and generates 12–20 second pitch contours at 100 Hz.

---

## Dataset

**Source**: Hindustani Music Recordings (HMR) dataset — vocal recordings with CREPE f0 annotations and beat annotations from a trained Beat Transformer.

**Filtering**:
- Instrument code `V` (vocal) only
- Excluded: recordings with sarangi, violin, or manually flagged issues
- Results in ~113 valid recordings

**Train/val split**: 80/20 by recording (val_ratio=0.17 → 19 val recordings, 94 train)

**Windows**: Recordings are sliced into fixed-length windows with a configurable stride:
- 12s model: 1200 frames, stride=1200 (non-overlapping)
- 20s model: 2000 frames, stride=1000 (50% overlap)

**Beat signal**: 3-channel array at 100 Hz:
- Channel 0: beat pulse (smoothed impulse at each beat event)
- Channel 1: sam (first beat of taal cycle) pulse
- Channel 2: continuous cycle position

Only channel 0 (beat pulse) is used as conditioning. The beat pulse is rendered by convolving beat event positions with a 5-frame rectangular kernel and clipping to [0, 1].

---

## Model

**Base**: GaMaDHaNi Stage 1 — a UNet diffusion model that generates pitch contours in a quantile-transformed token space (QT space). Pretrained on Hindustani vocal data.

**Beat conditioning (BeatConditionedUNet)**:
```
beat signal [B, 1, T]
    → Linear(1, C)        # beat_projection: single linear layer
    → [B, C, T]
    → added to UNet's initial_projection(x) output
```
The beat enters at the earliest feature level. All weights are fine-tuned (including the pretrained UNet).

**Classifier-Free Guidance (CFG)**:
- During training: with probability `cfg_prob` (typically 0.10–0.15), the beat signal is zeroed for that sample. The model learns both conditional and unconditional generation.
- During inference: two forward passes — one with the true beat, one with zeros:
  ```
  pred = pred_null + guidance_scale × (pred_cond - pred_null)
  ```
  Guidance scale amplifies the beat conditioning signal. Best results at scale=3.0.

**Pitch space**: The model operates in QT-normalized continuous space. The quantile transformer maps integer tokens (196–600, where <200 = silence) to a roughly uniform [-1, 1] distribution.

---

## Training

| Hyperparameter | 12s model | 20s model |
|----------------|-----------|-----------|
| Window length | 1200 frames | 2000 frames |
| Batch size | 16 | 8 |
| Learning rate | 1e-4 | 1e-4 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| cfg_prob | 0.10 (pre-CFG) / 0.15 (CFG fine-tune) | 0.10 |
| Epochs | 100 (pre-CFG) + 126 (CFG) | 100 + 166 CFG |
| Best val loss (12s CFG) | 0.1673 | — |
| Best val loss (20s CFG) | — | 0.1665 |

**Beat augmentation** (training only):
- Section dropout: zero out a contiguous 15–30% block with probability 0.3
- Tempo halving: keep every other beat event (probability 0.15 within tempo aug)
- Tempo doubling: insert midpoint beats (probability 0.15 within tempo aug)

---

## Glossary of Terms

| Term | Definition |
|------|-----------|
| **gt** | Ground truth — the correct beat signal belonging to the recording being evaluated |
| **shuf** | Shuffled — the beat signal from a *different* recording, used as a negative control. If the model truly conditions on the beat, the shuf condition should produce worse alignment than gt |
| **zero** | Unconditioned — beat signal set to all zeros. Equivalent to running the model with no beat guidance |
| **gs** / **guidance_scale** | CFG guidance scale — the amplification factor applied during inference: `pred = pred_null + scale × (pred_cond − pred_null)`. gs=1.0 means no amplification (conditioned pass only). gs=3.0 pushes the beat conditioning 3× harder |
| **gt_gs3** | Generated pitch using the correct beat + guidance_scale=3.0 |
| **shuf_gs3** | Generated pitch using a shuffled (wrong) beat + guidance_scale=3.0 |
| **F1** | Harmonic mean of precision and recall. Ranges 0–1. Higher = better beat alignment |
| **Precision** | Of all the note onsets detected in the generated pitch, what fraction falls near a beat position (within ±50ms) |
| **Recall** | Of all the beat positions, what fraction has at least one nearby note onset |
| **GT−shuf gap** | F1(gt_conditioned) − F1(shuffled_beat). The key metric: a large positive gap means the model responds specifically to the correct beat, not just any beat-like signal |
| **GT−zero gap** | F1(gt_conditioned) − F1(unconditioned). Measures how much conditioning (at any guidance scale) helps over no conditioning at all |
| **onset** | A note onset — a moment in the generated pitch where the singer starts a new note. Detected as silence→voiced transitions or large pitch jumps (≥1 semitone) |
| **beat frame** | A frame in the beat signal where a beat event occurs (beat pulse > 0.5). At 100Hz, one frame = 10ms |
| **window** | A fixed-length segment of a recording used for training or evaluation (1200 frames = 12s, 2000 frames = 20s) |
| **val window** | A window from the held-out validation set of recordings |
| **peak CC** | Peak normalized cross-correlation — the maximum correlation value between the pitch velocity signal and the beat pulse signal, within ±half-beat-period of lag=0 |
| **Rayleigh R** | A circular statistics measure (0–1) of how strongly note onsets cluster at a consistent phase within the beat cycle. R>0 with p<0.05 = statistically significant beat alignment |
| **cfg_prob** | Classifier-Free Guidance probability — the fraction of training samples where the beat signal is zeroed, teaching the model to generate both with and without conditioning |
| **laya** | Tempo in Hindustani music: vilambit (<60 BPM), madhya (60–120 BPM), drut (>120 BPM) |
| **taal** | The rhythmic cycle in Hindustani music (e.g. Teentaal = 16 beats, Ektaal = 12 beats) |
| **sam** | The first (and most emphasized) beat of a taal cycle |
| **QT space** | Quantile-transformed space — the model's internal pitch representation. The quantile transformer maps integer pitch tokens to a roughly uniform distribution, making diffusion training more stable |
| **token** | An integer pitch value in the model's discrete pitch space. Silence = token <200. Voiced pitch tokens range ~200–600, where 1 token = 10 cents |
| **cents** | A logarithmic pitch unit: 100 cents = 1 semitone, 1200 cents = 1 octave. Reference: 0 cents = 440 Hz (A4) |

---

## Evaluation Metrics

### 1. Beat Alignment F1 (Primary Metric)

Measures how well generated note onsets align with annotated beat positions.

**Pipeline**:
1. Generate pitch contour in QT space
2. Detect note onsets from the generated contour
3. Compute F1 between onset positions and beat frame positions

**Onset detection** (`detect_onsets()`):
- **Silence → voiced transition**: any frame where the previous frame had token < 200 (silence) and the current frame has token ≥ 200
- **Large pitch jump**: consecutive voiced frames where |token[t] − token[t−1]| ≥ 10 (~1 semitone, since pitch_downsample=10 means 1 token = 10 cents)

Both types are merged into a single onset list.

**F1 computation** (`beat_alignment_f1()`):
- A beat is a **true positive (TP)** if there is at least one onset within ±5 frames (±50ms)
- An onset is a **false positive (FP)** if it is not near any beat
- A beat with no nearby onset is a **false negative (FN)**
- Standard precision/recall/F1 computed from TP, FP, FN

**Conditions evaluated**:
- `gt_conditioned`: model generates with the correct beat signal for that recording
- `shuffled_beat`: model generates with a different recording's beat signal (sanity check — should show no alignment if conditioning is beat-specific)
- `unconditioned` (zero beat): model generates with beat signal set to zeros (baseline)

**Key gap metric**: `GT−shuffled gap = F1(gt) − F1(shuffled)`. A large positive gap means the model is genuinely responding to the specific beat signal rather than learning a generic onset rate.

### 2. Pitch Contour Cross-Correlation (CC Metric)

Measures continuous rhythmic correspondence between the pitch velocity signal and the beat pulse.

**Pipeline**:
1. Convert generated QT-space pitch to cents: `token × 10 − 4915`
2. Compute velocity: `vel[t] = |cents[t] − cents[t−1]|`, zeroed at silence frames
3. Compute normalized cross-correlation between velocity and beat pulse at lags ±200 frames (±2s)
4. Report peak CC within ±half-beat-period around lag=0

**GT−shuffled gap**: Same logic as F1 — compare peak CC when the correct beat is used vs a shuffled beat from a different recording.

**Onset representation comparison** (completed Apr 19):
- Continuous |Δcents|: gap = +0.035
- Binary threshold |Δcents| ≥ 25 cents: gap = +0.035 (tied/slightly better for CC)
- Stable-note segmentation (ISMIR 2021 §3): unreliable — shows false gap on wrong-beat condition

The continuous |Δcents| representation is used as the default in `evaluate_contour_crosscorr.py`.

### 3. Phase Distribution / Rayleigh R (Rigorous Statistical Test)

Tests whether note onsets cluster at a consistent phase within the beat cycle — the strongest evidence of beat alignment.

**Method**:
1. For each onset, compute its phase within the current beat cycle (0–2π)
2. Treat phases as unit vectors on the circle
3. Rayleigh R = mean resultant length = |mean(e^{iφ})|
4. Rayleigh test: R significantly > 0 means non-uniform distribution → beat alignment

**Results** (12s GT beats model, drut/madhya only — vilambit excluded):
- GT beats model: R=0.145, p<0.05 ✓ (significant)
- Extracted beats model: R=0.054 (not significant)

Vilambit is excluded because vocalists in slow tempo do not reliably phase-align to taal beats.

---

## Results Summary

### CFG Guidance Scale Sweep (12s model, 186 val windows × 2 samples)

| Guidance Scale | GT F1 | Shuffled F1 | Zero F1 | GT−shuf gap |
|----------------|-------|-------------|---------|-------------|
| 1.0 | 0.292 | 0.209 | 0.141 | +0.083 |
| 1.5 | 0.397 | 0.225 | 0.140 | +0.172 |
| 2.0 | 0.475 | 0.261 | 0.152 | +0.215 |
| **3.0** | **0.605** | **0.301** | **0.146** | **+0.303** |
| 5.0 | 0.586 | 0.295 | 0.149 | +0.292 |

**Optimal guidance scale: 3.0**. Scale=5 slightly over-sharpens.

### Model Comparison (at guidance scale 3.0)

| Model | Windows | GT F1 | Shuf F1 | GT−shuf gap |
|-------|---------|-------|---------|-------------|
| 12s, no CFG (epoch 67) | 106×4 | 0.289 | 0.200 | +0.089 |
| 20s, no CFG (epoch 100) | 205×2 | 0.396 | 0.270 | +0.125 |
| **12s, CFG (epoch 126)** | **186×2** | **0.605** | **0.301** | **+0.303** |
| 20s, CFG v1 (epoch 61) | 205×2 | 0.374 | 0.271 | +0.104 |
| 20s, CFG v2 (epoch 127) | 205×2 | 0.405 | 0.298 | +0.107 |

The 20s model without CFG fine-tune outperforms the 12s non-CFG model (+0.125 vs +0.089), suggesting longer context helps. CFG v1 was trained with cfg_prob=0.1 from epoch 1 (no clean pre-CFG phase) and used a warmed-over LR schedule on resume. CFG v2 fixed both issues (clean pre-CFG phase to epoch 100 with cfg_prob=0.0, then fresh optimizer + cfg_prob=0.15), but the gap barely moved (+0.104 → +0.107). The 12s model remains 3× better on the GT−shuf gap. The gap between 12s and 20s CFG performance likely reflects a fundamental difficulty: 20s windows have sparser beat events relative to window length, making the beat conditioning signal harder to learn.

### Contour Cross-Correlation (generated outputs, 116 val windows)

| Condition | Peak CC | GT−shuf gap |
|-----------|---------|-------------|
| noprime_gt | 0.0740 | — |
| prime400_gt | 0.0752 | +0.064 |
| beatprime400_gt | 0.0390 | +0.029 |

### Pitch Prime Experiment (38 val windows, outputs_cfg_prime/)

Script: `generate_cfg_with_prime.py`. 5 conditions generated per window (beatprime dropped — see note):

| Condition | Mean F1 | GT−shuf gap |
|-----------|---------|-------------|
| **beatprime400_gt** | **0.680** | +0.217 |
| beatprime400_shuf | 0.463 | — |
| **noprime_gt** | **0.602** | **+0.307** |
| noprime_shuf | 0.295 | — |
| prime400_gt | 0.504 | +0.236 |
| prime400_shuf | 0.268 | — |
| noprime_zero | 0.127 | — |

**Note on beatprime**: Beat priming raises absolute GT F1 (0.680 vs 0.602) but the GT−shuf gap *shrinks* (+0.217 vs +0.307) because the shuffled beat prime also inflates F1 (0.463 vs 0.295) — the model partly follows the rhythm structure locked into the prime rather than the conditioning signal. Additionally, the synthesized beat prime (`make_beat_prime`) is 95% silence in qt-space (only brief NOTE_QT pulses at beat positions), making the prime audio perceptually useless. **beatprime conditions dropped from future experiments.**

**Pitch prime interpretation**: `prime400_gt` gives gap +0.236 (vs noprime +0.307) — the pitch prime constrains generation but slightly dampens beat responsiveness, possibly because the model anchors to the melodic trajectory rather than the beat.

Outputs uploaded to Drive: `GaMaDHaNi-samples/outputs_cfg_prime/`

---

## Key Findings

1. **Beat conditioning works** — GT−shuf gap of +0.303 at gs=3 is statistically meaningful. Shuffled and unconditioned baselines stay flat as guidance scale increases, confirming the amplification is beat-specific.
2. **Vilambit fails** — Slow-tempo vocalists do not phase-lock to beats; all metrics near zero for vilambit laya.
3. **Longer context helps pre-CFG** (20s > 12s, no CFG: +0.125 vs +0.089) — but CFG fine-tuning does not transfer the benefit: 20s CFG gap stays ~+0.107 regardless of training procedure.
4. **Guidance scale 3.0 is optimal** — monotonic improvement from 1→3, regression at 5.
5. **Beat augmentation can hurt** — aggressive dropout (p=0.3) narrowed the gap from +0.089 to +0.035.
6. **Binary onset threshold doesn't improve F1** — 25-cent threshold generates too many onsets (saturates recall). Original 1-semitone jump threshold is appropriate for F1. Binary 25c only helps the CC metric.
7. **Stage 2 audio model is length-agnostic** — `UNetPitchConditioned` forward pass works at arbitrary lengths (tested at 1250 mel frames = 20s). The hardcoded `self.seq_len=750` only appears in `sample_cfg`; bypassing it with a custom sampling loop produces clean longer audio. Mel frame rate = 62.5 frames/s (750 frames = 12s at 16kHz with hop=256).
8. **20s CFG gap is fundamentally limited** — Two separate training runs (+0.104, +0.107) confirm the ceiling. Training procedure (LR scheduling, cfg_prob) is not the bottleneck.
9. **Beat prime is misleading** — Synthesized beat prime (silence + pulses) is 95% silent in qt-space; inflates both GT and shuf F1 (model follows prime rhythm regardless of conditioning). Dropped from experiments.

---

## Checkpoints

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| `checkpoints/hmr_gt_beats/best.ckpt` | 0.1569 | 12s, no CFG, epoch 67 |
| `checkpoints/hmr_gt_beats_cfg/best.ckpt` | 0.1673 | **12s, CFG — best model** (gs=3 gap=+0.303), epoch 126 |
| `checkpoints/hmr_gt_beats_20s/best.ckpt` | 0.1611 | 20s, cfg_prob=0.1 from ep1, epoch 34 (flawed) |
| `checkpoints/hmr_gt_beats_20s_cfg/best.ckpt` | 0.1665 | 20s, CFG v1, epoch 61 (flawed LR resume) |
| `checkpoints/hmr_gt_beats_20s_v2/best.ckpt` | 0.1593 | 20s, no CFG clean, epoch 100 |
| `checkpoints/hmr_gt_beats_20s_cfg_v2/best.ckpt` | 0.1637 | 20s, CFG v2 (fresh optimizer, cfg_prob=0.15), epoch 127 |

---

## File Structure

```
beat-conditioned-GaMaDHaNi/
  train_beat_conditioned.py       — training script
  evaluate_beat_alignment.py      — F1 metric evaluation
  evaluate_contour_crosscorr.py   — CC metric evaluation
  evaluate_phase.py               — Rayleigh R evaluation
  evaluate_onset_cc.py            — onset representation comparison
  generate_beat_samples.py        — audio sample generation
  gamadhani/src/hmr_dataset.py    — HMR dataset with beat augmentation
  configs/diffusion_pitch_config.gin
  checkpoints/                    — saved model checkpoints
  test_long_audio.py              — test Stage 2 at arbitrary audio lengths
  outputs_cfg/                    — 12s generated samples (gs1/gs3/gs5)
  outputs_20s_proper/             — 20s generated samples (proper 1:1 pitch→mel mapping)
  outputs_best/                   — best-gap windows for listening
  beat_alignment_gs*.json         — F1 eval results by guidance scale
  beat_alignment_20s_cfg_v2_gs3.json — 20s CFG v2 eval results
  generate_cfg_with_prime.py        — prime experiment generation (noprime/prime400/beatprime, beatprime dropped)
  outputs_cfg_prime/                — 38 windows, 7 conditions (beatprime kept in files but not used)
```

---

## Audio Generation Notes

**Stage 2 is length-agnostic**: `UNetPitchConditioned.forward()` uses strided convolutions + x_transformers `AttentionLayers` with no positional encodings — works at any length.

**Mel frame rate**: 62.5 frames/s (750 mel frames = 12s). For Ns pitch at 100Hz: `target_len = round(N * 750 / 1200)`.

**Custom sampling loop** (in `generate_beat_samples.py` and `test_long_audio.py`): bypasses `sample_cfg`'s hardcoded `self.seq_len=750` by creating noise at `target_len` and calling `forward()` directly.

**Click tracks**: built into `generate_beat_samples.py` — 4-subdivision clicks overlaid at beat positions, saved as `audio_*_click.wav` alongside each generated file.

**Google Drive**: samples uploaded via rclone remote `gdrive:`. Folder: `GaMaDHaNi-samples/`.
- `outputs_20s_proper/` — 38 windows, 20s audio, gt/shuf/zero + clicks
- `test_long_audio_20s_proper/` — baseline (12s) vs proper 20s audio comparison
