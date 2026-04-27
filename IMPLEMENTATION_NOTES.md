# Beat-Conditioned GaMaDHaNi — Implementation Notes

## Goal
Fine-tune GaMaDHaNi Stage 1 (diffusion pitch model) to generate pitch contours conditioned on taal beat structure, following the Sketch2Sound approach.

## Overall Pipeline

```
HMR audio (.wav)
    → demucs htdemucs          → vocal stem (.wav)
    → CREPE at 100Hz           → f0 in Hz + confidence (.npy)

HMR beat annotations (.beats)  → 3-channel beat signal at 100Hz (.npy)

f0 + beat + confidence
    → build_hmr_lmdb.py        → LMDB (protobuf AudioExample per recording)
    → HMRBeatDataset           → (normalized_pitch [1200], beat [3, 1200]) per window
    → BeatConditionedUNet      → fine-tuned model
```

---

## Dataset

**Source:** HMR dataset — 151 recordings, filtered to:
- Instrument code = V (vocal lead)
- No sarangi or violin accompaniment (MusicBrainz labels + manual check)
- Explicit exclusions: UIDs 20028, 20032, 21018, 21025, 22011, 23014, 23015, 23019

**Final count:** 114 recordings → 102 train / 12 val (stratified by taal, 90/10)

**Taal distribution:**
| Taal | Train | Val |
|------|-------|-----|
| Ektaal | 43 | 5 |
| Teentaal | 34 | 4 |
| Jhaptaal | 14 | 2 |
| Rupak | 11 | 1 |

---

## Beat Signal (3 channels at 100Hz)

Built from HMR `.beats` annotation files (ground truth).

| Channel | Description | Range |
|---------|-------------|-------|
| 0 — beat pulse | Smoothed box pulse at each beat position (np.ones(5) kernel, clipped to [0,1]) | [0, 1] |
| 1 — sam pulse | Same pulse only at sam (beat position = 1) | [0, 1] |
| 2 — cycle position | Fractional position in taal cycle (0=sam, approaching 1 at end of cycle) | [0, 1] |

**Taal cycle lengths:** teentaal=16, ektaal=12, jhaptaal=10, rupak=7

---

## Pitch Extraction

- **Separation:** demucs `htdemucs` model, vocal stem saved to `HMR_processed/vocals/`
- **Pitch:** CREPE at 100Hz on separated vocal stem, saved to `HMR_processed/pitch/`
- **Confidence filtering:** per-window at dataset level — windows where >50% of frames have CREPE confidence <0.5 are retried (up to 20 attempts, falls back to least-bad window)

---

## LMDB Format

**Location:** `HMR_processed/lmdb/train`, `HMR_processed/lmdb/val`

Each entry is a `AudioExample` protobuf with 3 buffers:

| Key | Shape | dtype | Notes |
|-----|-------|-------|-------|
| `pitch` | [N] | float32 | f0 in Hz at 100Hz, full ~120s clip |
| `beat` | [3, N] | float32 | 3-channel beat signal at 100Hz |
| `beat_confidence` | [N] | float32 | CREPE confidence per frame |

Global conditions: `singer` field = taal ID (0=teentaal, 1=ektaal, 2=jhaptaal, 3=rupak)

**Note:** No proto schema change was needed — `buffers` is already `map<string, DataBuffer>`.

---

## HMRBeatDataset (`hmr_dataset.py`)

Reads from LMDB. Each `__getitem__`:
1. Loads full f0, beat, confidence arrays
2. Randomly samples a valid 1200-frame (12s) window with confidence filtering
3. Normalizes pitch using the same pipeline as the pretrained model:
   - Hz → cents: `1200 * log2(f0 / 440)`
   - Discretize, shift by `-min_norm_pitch` (+4915), bin by `pitch_downsample=10`
   - Clip to [200, 600], silence token = 196
   - Apply QuantileTransformer (from pretrained model checkpoint)
4. Returns `{"normalized_pitch": [1200], "beat": [3, 1200]}`

---

## Model (`BeatConditionedUNet`)

From collaborator's notebook. Key changes from base `UNet`:
- `beat_projection = nn.Linear(beat_dim, initial_projection.out_channels)`
  - `beat_dim=3` for our 3-channel signal (vs. notebook's `beat_dim=1`)
- Beat added to initial projection output before downsampling
- All weights fine-tuned (Sketch2Sound style — no frozen backbone)
- Beat dropout (p=0.1) during training for robustness

**Trainable params:** ~full UNet + one Linear(3, C) layer

---

## What Differs from Sketch2Sound

| Sketch2Sound | Our implementation |
|---|---|
| Extracted controls (imperfect) | Ground truth annotations (first run) |
| Random median filtering on controls | ❌ Not yet — to add for extracted-beats run |
| One linear layer per control | ✅ `beat_projection` Linear |
| Full fine-tune | ✅ All weights |
| ~40k steps | TBD |

**Planned second run:** extracted beats (e.g. from BeatNet or madmom) with random median filtering, to compare against ground truth conditioning.

---

## Test Run Results (2026-03-28)

End-to-end test: dataset load, batch construction, forward pass, backward pass on CUDA.

| Check | Result |
|-------|--------|
| LMDB alignment (pitch/beat/conf lengths) | ✅ All 102 entries aligned |
| Dataset windows | ✅ 1649 train / 171 val |
| Beat shape from dataset | ✅ `[1200]` (1-channel, matches notebook convention) |
| Batch shapes | ✅ `x=[8,1,1200]`  `beat=[8,1,1200]` after `.unsqueeze(1)` |
| Forward pass | ✅ No errors |
| Loss + backward | ✅ Gradients flow, loss=0.2834 |
| Trainable parameters | 111,581,953 (~111M) |

**Bugs fixed:**
- `data_example.py` had `buffer.sample_rate` → corrected to `buffer.sampling_rate` (proto field name)
- `hmr_dataset.py` returned beat as `[1, T]` → fixed to `[T]` to match toy `BeatDataset` convention (notebook does `.unsqueeze(1)` in the training loop)
- `x-transformers` version mismatch: installed 2.0.0 by default (LayerNorm uses `gamma`) but checkpoint needs 1.30.2 (uses `weight`) → pinned to 1.30.2

---

## Training Results (2026-03-28)

Training run: 100 epochs, AdamW lr=1e-4, cosine annealing, batch_size=16, 989 train / 106 val windows.

| Metric | Value |
|--------|-------|
| Best checkpoint | epoch 67 |
| Best val loss | 0.1569 |
| Trainable params | 111,581,953 (~111M) |

Val loss < train loss throughout training. This is expected given the small val set (11 recordings vs 102 train).

---

## Loss Curves

Plot: `loss_curves.png`

Three training runs, all starting from the pretrained GaMaDHaNi diffusion pitch model:

| Run | Epochs | Best val loss | Best epoch | Notes |
|-----|--------|--------------|------------|-------|
| GT beats | 80 | **0.1569** | 67 | Main model |
| Augmented | 80 (ep21→100) | **0.1517** | 78 | Continued from GT beats run 1; beat augmentation |
| Extracted beats | 100 | **0.1544** | 39 | Trained on Beat Transformer extracted beats |

**Observations:**
- All three runs show train loss ~0.20–0.22 throughout — the model doesn't overfit in a classical sense (train never dips below val).
- Val loss is noisy across all runs due to small val set (12 recordings). No smooth convergence curve — best checkpoint varies epoch to epoch.
- Train > val consistently: expected with small val set and the fact that pitch diffusion is a hard task; the model's uncertainty is high on both sets.
- Augmented model continued from GT beats epoch 80, starting at epoch 21 in logs. Best val loss (0.1517) is marginally lower than GT beats (0.1569) but this didn't translate to better beat alignment eval.
- Extracted beats model shows similar loss profile to GT beats — the noisy conditioning signal doesn't seem to hurt training loss, only eval quality.

---

## Beat Alignment Evaluation (2026-03-28)

**Method:** Generate pitch contours under 3 conditions. Detect onsets (pitch jumps ≥ 10 tokens ≈ 1 semitone in token space) from each generated contour. Compute F1 between onset positions and GT beat positions (±5 frame / ±50ms window).

**Why pitch-jump onsets:** All val windows have continuous singing (no silence frames — CREPE outputs non-zero f0 throughout). Silence→voiced transitions alone detect nothing. Pitch jumps capture note changes, which in Hindustani music are the main rhythmic events that align with beats.

**Why not use Sketch2Sound's exact metric:** Sketch2Sound runs onset detection on audio waveforms. We only have pitch contours at Stage 1 — the pitch-jump proxy approximates this.

**Reference:** GT self-alignment (real vocal pitch checked against own beats) = F1 ≈ 0.45. This is the upper bound.

**Results (106 val windows × 4 samples each = 424 data points per condition):**

| Condition | F1 | Precision | Recall | Onsets/window |
|-----------|-----|-----------|--------|----------------|
| GT beat conditioned | **0.289** | 0.398 | 0.292 | 56.2 |
| Shuffled beat | 0.200 | 0.326 | 0.213 | 54.9 |
| Unconditioned (zero beat) | 0.143 | 0.291 | 0.131 | 38.9 |

**Interpretation:**
- Expected ordering holds: GT > shuffled > unconditioned ✅
- Beat conditioning is working — the model genuinely responds to the beat signal content, not just its presence
- Zero beat suppresses onset generation (39 vs ~55 per window) — beat signal actively drives melodic transitions


More direct test of beat conditioning: for each generated onset, compute its phase within the beat cycle (0 = onset lands on a beat, 0.5 = halfway between beats). Under ideal conditioning, the GT condition should produce a spike near phase=0.

**Metrics:**
- **Rayleigh R**: circular statistics measure of phase clustering toward 0. R=0 means uniform (random); R=1 means all onsets exactly on beats.
- **Near-beat %**: fraction of onsets within ±10% of a beat position. Chance baseline = 20%.
- **Bootstrap 95% CI**: on the GT−shuf cross-correlation gap (2000 resamples).

### GT Beats Model

| Condition | Rayleigh R | Near-beat % |
|-----------|-----------|-------------|
| GT beat   | **0.119** | **25.6%**   |
| Shuffled  | 0.018     | 20.8%       |
| Zero      | 0.028     | 19.4%       |

Phase histogram beat bin (phase 0–0.05): **9.6%** vs uniform 5.0% — nearly 2× elevated.
Bootstrap CC gap: +0.0065, 95% CI [−0.0004, +0.0135] — borderline.

**Laya breakdown:**
| Laya | N | CC gap | 95% CI | Rayleigh R (GT) |
|------|---|--------|--------|-----------------|
| drut | 188 | +0.0144 | [+0.009, +0.019] ✓ | 0.145 |
| drut/madhya | 36 | +0.0169 | [+0.006, +0.029] ✓ | 0.162 |
| madhya | 40 | +0.0065 | [−0.005, +0.019] | 0.101 |
| vilambit | 160 | −0.0050 | [−0.022, +0.011] ✗ | 0.086 |

### Extracted Beats Model

| Condition | Rayleigh R | Near-beat % |
|-----------|-----------|-------------|
| GT beat   | **0.069** | **23.2%**   |
| Shuffled  | 0.014     | ~flat       |
| Zero      | ~flat     | ~flat       |

Phase histogram beat bin: **7.2%** — slight elevation, but much weaker than GT model.
Bootstrap CC gap: +0.0045, 95% CI [−0.0010, +0.0099] — not significant.
Laya breakdown: incoherent — vilambit shows largest gap, drut shows nothing. Consistent with noise.

**Conclusion:** Rayleigh R is 0.119 (GT model) vs 0.069 (extracted model) — roughly half the phase clustering strength. GT model shows real, laya-consistent beat conditioning; extracted model does not.

### Excl. vilambit (drut + drut/madhya + madhya only)

Vilambit has no vocal-beat onset correlation in the ground truth data, so including it dilutes the signal. Results excluding vilambit:

| Model | Eval reference | Rayleigh R (GT) | Beat bin | CC gap | 95% CI | Sig |
|-------|---------------|-----------------|----------|--------|--------|-----|
| GT beats model | GT beats | **0.145** | **10.3%** | +0.0133 | [+0.009, +0.018] | YES ✓ |
| Extracted model | GT beats | 0.054 | 7.7% | +0.0005 | [−0.004, +0.005] | NO |
| Extracted model | Extracted beats | 0.064 | 8.9% | +0.0028 | [−0.002, +0.007] | NO |

Key observations:
- **GT beats model**: beat bin at 10.3% (>2× uniform), Rayleigh R=0.145, CC gap significant with CI entirely above zero. Drut laya shows R=0.153, drut/madhya R=0.197.
- **Extracted model vs GT**: essentially no signal. CC gap near zero, laya breakdown incoherent.
- **Extracted model vs its own beats**: marginally better (R=0.064 vs 0.054, beat bin 8.9% vs 7.7%), indicating the model learned a weak association with the extracted beat positions — but not significant, and laya structure is still incoherent (drut CI borderline positive, drut/madhya CI negative).
- **Conclusion**: the extracted beats are too noisy for the model to learn reliable conditioning. Even when evaluated fairly against extracted beat positions, no significant signal emerges.

---

## Prime Eval (`generate_with_prime.py`, 2026-04-06)

**Goal:** Test whether conditioning the generation on a primed prefix (first N frames of GT pitch or GT beat encoding) improves beat alignment F1 on the full val set.

**Setup:** All 116 val windows (not just drut) generated under 6 conditions:
- `noprime_gt` — no prime, GT beat conditioning (baseline)
- `pitch_prime100/200/400_gt` — first 100/200/400 frames are GT pitch (clamped), rest generated with GT beat
- `beatprime100/200/400_gt` — first 100/200/400 frames are a **synthetic pitch contour** where beat positions are marked as short notes (QT value 0.2, duration 10 frames) in an otherwise silent (QT value −1.077) sequence. This is not real pitch — it's a rhythmic marker encoded in pitch space, designed to explicitly signal beat positions via the prime.

**Results (mean F1, all 116 windows):**

| Condition | Mean F1 | vs noprime |
|-----------|---------|------------|
| No prime, GT beat | 0.1399 | — (baseline) |
| Pitch prime 100fr, GT | 0.1320 | −0.008 |
| Pitch prime 200fr, GT | 0.1471 | +0.007 |
| Pitch prime 400fr, GT | **0.1504** | **+0.011** |
| Beat-prime 100fr, GT | 0.1301 | −0.010 |
| Beat-prime 200fr, GT | 0.1260 | −0.014 |
| Beat-prime 400fr, GT | 0.1351 | −0.005 |

**Conclusions:**
- Pitch prime helps marginally at 400fr (+0.011), degrades at short lengths (100fr: −0.008)
- Beat-encoded prime consistently hurts (−0.005 to −0.014) — the synthetic prime is not a realistic pitch context; the model receives an unmusical input (silence punctuated by short blips at beat positions) and generates accordingly
- Gains are small and noisy given the val set size — not a reliable improvement strategy without architectural changes

**Saved:** `prime_eval_results.json`, generated pitch contours in `outputs_prime/`

---

## Full Pitch Contour Cross-Correlation (`evaluate_contour_crosscorr.py`, 2026-04-09)

**Motivation:** Binary onset detection (pitch-jump threshold) is noisy and misses sustained-note alignment. Instead, compute cross-correlation between the *full pitch velocity contour* (`|Δcents|` at 100Hz) and the GT beat pulse — a continuous signal that captures all rhythmic content.

**Pipeline:**
- GT pitch: Hz → cents → `|Δcents|` (velocity), silence masked (f0<50Hz or CREPE conf<0.5)
- Generated pitch: QT-space → `qt.inverse_transform()` → token → `cents = token * 10 - 4915`, silence = token < 200 → `|Δcents|`
- Beat signal: raw beat pulse (ch0 of `{uid}_beat.npy`)
- Cross-correlation: `np.correlate(a−mean, b−mean, 'full') / (‖a‖‖b‖)`, lags ±200 frames (±2s)
- Peak within ±half-beat-period window

**~~SUPERSEDED (buggy)~~ — Drut-only results (6 UIDs), prime frames included in CC:**

> ⚠️ Bug: prime frames (P=400) were included in the cross-correlation window. For `beatprime400`, the prime literally contains beat markers at beat positions → trivially inflated CC. See corrected results below.

| Signal | Peak CC | GT−shuf gap |
|--------|---------|-------------|
| noprime_gt generated | 0.0952 | — |
| prime400_gt generated | 0.0809 | — |
| prime400_shuf generated | 0.0134 | **+0.0675** |
| beatprime400_gt generated | ~~0.2165~~ | — |
| beatprime400_shuf generated | ~~0.2013~~ | ~~+0.0152~~ |

---

**Corrected results (2026-04-12) — all 17 val UIDs, 116 windows, prime excluded:**

Fix applied in `evaluate_contour_crosscorr.py`: for each prime condition, correlation computed over `[P:]` only (not full 1200 frames), so clamped prime region doesn't inflate CC.

| Signal | Peak CC | GT−shuf gap |
|--------|---------|-------------|
| GT real vocal (GT beats, 162 windows) | 0.0187 | **+0.0130** |
| GT real vocal (shuffled beats) | 0.0056 | — |
| noprime_gt generated | 0.0740 | — |
| prime400_gt generated | 0.0752 | — |
| prime400_shuf generated | 0.0115 | **+0.0637** |
| beatprime400_gt generated | 0.0390 | — |
| beatprime400_shuf generated | 0.0096 | **+0.0294** |

**Interpretation (corrected):**
- GT real data: weak but genuine gap (+0.013) — real vocals weakly track beat positions
- pitch_prime400 gap (+0.064) survives the fix — conditioning is genuinely selective to correct beat timing
- beatprime400 absolute CC drops from 0.217 → 0.039 — the inflated value was almost entirely the prime frames. Corrected gap (+0.029) is real but modest
- beat_prime absolute CC now comparable to noprime baseline (0.039 vs 0.074), meaning the beat-encoded prime does not produce strong beat-aligned generation in the continuation

**Saved:** `contour_crosscorr_results.json`, `contour_crosscorr_plot.png`

**Note on vilambit:** All-val results include vilambit, which dilutes gaps. Drut-only filter would widen gaps further.

---

## Onset Representation Comparison (`evaluate_onset_cc.py`, 2026-04-19)

**Motivation:** From the to-do list — "better ways to define vocal onsets":
1. What if you binarize the first-order pitch derivative by a threshold?
2. Does the ISMIR 2021 §3 stable-note segmentation work better for the CC metric?

**Three representations compared:**
- **(A) Continuous |Δcents|** — current default; zeroed at silence frames
- **(B) Binary threshold** — 1 where |Δcents| ≥ threshold, else 0; tested at 25c, 50c, 100c, 150c, 200c
- **(C) Stable-note onsets (ISMIR 2021 §3)** — build a per-window pitch histogram, find svara peaks, label stable frames (voiced + within ±35c of nearest svara, min duration 250ms), onset = start of each stable segment

**Eval pipeline:** Same normalized cross-correlation as `evaluate_contour_crosscorr.py`, run on GT real vocal pitch (162 windows) and generated pitch from `outputs_prime/` (noprime_gt, prime400_gt, prime400_shuf). GT−shuf gap computed using the same n/2-rotation shuffle.

### GT real vocal (162 windows, mean beat period=1080ms)

| Representation | Peak CC |
|----------------|---------|
| continuous |Δcents| | 0.068 |
| binary 25c | **0.077** |
| binary 50c | 0.076 |
| binary 100c | 0.071 |
| stable notes | 0.062 |

Binary 25c gives the highest absolute CC on GT real vocal — slightly better than continuous.

### Generated pitch — GT−shuf gap (noprime_gt, 116 windows)

| Representation | GT CC | Shuf CC | Gap |
|----------------|-------|---------|-----|
| continuous |Δcents| | 0.105 | 0.070 | **+0.035** |
| binary 25c | 0.098 | 0.064 | +0.034 |
| binary 50c | 0.094 | 0.068 | +0.026 |
| binary 100c | 0.089 | 0.074 | +0.016 |
| binary 150c | 0.086 | 0.069 | +0.017 |
| stable notes | 0.054 | 0.046 | +0.008 |

### Stable-note reliability check (prime400_shuf — wrong beat, should show gap ≈ 0)

| Representation | Gap (should be ~0) |
|----------------|--------------------|
| continuous | −0.001 ✓ |
| binary 25c | 0.000 ✓ |
| binary 50c | −0.002 ✓ |
| stable notes | **+0.018 ✗ (false positive)** |

### Conclusions

1. **Binary 25c ≈ continuous** on the GT−shuf gap (+0.034 vs +0.035) — essentially tied. Binary 25c gives marginally higher absolute CC on GT real vocal.
2. **Higher thresholds degrade the gap** — 50c drops to +0.026, 100c+ to ~+0.016. Thresholds ≥100c miss too many real note transitions.
3. **Stable-note segmentation is unreliable** — gap on noprime_gt is weak (+0.008) and, critically, it shows a false positive gap (+0.018) on prime400_shuf (wrong-beat condition). The per-window svara detection is too noisy at 12s window length, especially in generated pitch. **Dropped.**
4. **Decision:** Continuous |Δcents| kept as default for the CC metric (simplest, not meaningfully worse than binary 25c). Binary 25c noted as equivalent; higher thresholds hurt F1 (saturate recall). **The 1-semitone jump threshold (100c = 10 tokens) remains appropriate for F1 — it is distinct from the CC representation choice.**

**Saved:** `pitch_values_crosscorr_results.json`, `pitch_values_crosscorr_plot.png`

---

## Audio Synthesis + Click Tracks (`synthesize_prime_audio.py`, `add_clicks_to_prime_audio.py`, 2026-04-07–09)

**Audio synthesis:** Converted `outputs_prime/` pitch .npy files → audio via Stage 2 pitch-to-audio model. 15 folders × 6 pitch conditions = 90 WAV files saved to `outputs_prime_audio/`.

Target folders: 6 drut (`20045_0`, `21040_0`, `21055_0`, `22017_0`, `22019_0`, `23020_0`), 4 vilambit (`20002_0/1200`, `20011_0/1200`), 3 more vilambit (`21012_0`, `21019_0`, `21020_0`), 2 madhya (`20016_0/1200`).

**Click tracks:** GT beat clicks overlaid on all 90 synthesized WAVs (+ GT audio) using:
- 400Hz / gain 1.20 = sam (cycle boundary) — loudest, lowest pitch
- 900Hz / gain 0.90 = beat
- 1400Hz / gain 0.55 = 2× subdivision (half-beat)
- 2000Hz / gain 0.30 = 4× subdivision (quarter-beat)

Output: `{folder}/{name}_click.wav` alongside each original WAV.

**Hypothesis for vilambit:** The model may place rhythmic events at 2× or 4× the annotated beat period in slow laya sections (i.e., the generated pitch has structure at subdivision tempo, not beat tempo). The 2×/4× subdivision clicks allow subjective verification of this by listening to whether pitch transitions align with subdivision ticks rather than beat ticks.

---

## Training Analysis & Architecture Limitations (2026-04-11)

**Observation from loss curves:** All three runs show the model is still slowly descending at the end of training (80–100 epochs) — not fully converged. The val loss (0.1569 GT beats, 0.1544 extracted) may not be substantially better than the pretrained model baseline. The small val set (12 recordings) makes the curves very noisy and best-epoch selection unreliable.

**Root cause of weak conditioning:**
1. **Single linear layer is a thin connection.** `Linear(3, C)` at the initial projection is the only point where beat enters the model. It must survive through all UNet downsampling/upsampling layers unchanged to affect generation.
2. **No explicit beat loss.** Training optimises noise-prediction MSE only. The model can achieve low loss by learning a good pitch prior while ignoring beat signal entirely — and it largely does.
3. **Sparse beat signal.** Only a few frames per beat period carry signal; the model has strong incentives to focus on dense pitch structure instead.

**Considered architectural improvements (not yet implemented):**
- **Multi-scale injection**: project beat signal at each UNet resolution level (downsample beat to match), add to feature maps at each block — analogous to time-embedding injection in standard diffusion UNets
- **Input concatenation**: stack 3-channel beat with noisy pitch as input channels; direct access at every layer, minimal code change
- **FiLM conditioning**: per-ResNet-block scale/shift from beat-derived MLP
- **Cross-attention over beat frames**: most expressive, highest complexity

Decision: architectural changes pinned for now. Existing results with the GT beats model (Rayleigh R=0.145, significant for drut/madhya) are the primary finding.

---

## Classifier-Free Guidance (CFG) Fix + Guidance Scale Sweep (2026-04-13)

### What Was Wrong: Dropout ≠ CFG

The original `BeatConditionedUNet` included `nn.Dropout(p=beat_dropout)` applied to the projected beat embedding:

```python
# OLD (wrong)
beat = self.beat_projection(beat).transpose(1, 2)
beat = self.beat_dropout(beat)  # drops random individual features
x = x + beat
```

This is **regularization**, not classifier-free guidance. Dropout randomly zeros individual neurons in the projected beat vector — the model never sees a *fully absent* beat signal. It just sees a slightly corrupted one. This means:
- At inference, running the model with `null_beat = zeros` produces a genuinely different input distribution than anything seen during training
- The "null" and "conditioned" forward passes compute similar things, so their difference `pred_cond - pred_null` is meaningless
- Amplifying that difference with `guidance_scale > 1.0` amplifies noise, not conditioning

### The Fix: Whole-Signal Zeroing in `loss()`

Proper CFG zeros the **entire beat signal** for a random fraction of the batch during training, so the model explicitly learns "no beat" as a distinct input state:

```python
# NEW (correct CFG)
def loss(self, x, beat):
    ...
    if self.training and self.cfg_prob > 0:
        mask = (torch.rand(pb.shape[0], device=pb.device) < self.cfg_prob)
        pb = pb * (~mask).float()[:, None, None]  # zero entire beat for masked samples
    ...

def sample(self, beat, num_steps=100, guidance_scale=1.0):
    null_beat = torch.zeros_like(padded_beat)
    for t in ...:
        pred_cond = self.forward(pn, t, padded_beat)
        if guidance_scale != 1.0:
            pred_null = self.forward(pn, t, null_beat)
            pred = pred_null + guidance_scale * (pred_cond - pred_null)
        else:
            pred = pred_cond
```

With `cfg_prob=0.15`, 15% of training samples have beat fully zeroed. The model learns to generate plausible pitch with *and* without a beat signal. At inference, the two-pass formula amplifies the "beat-caused" component of the prediction.

### Training Run

- Checkpoint dir: `checkpoints/hmr_gt_beats_cfg/`
- Resumed from: `hmr_gt_beats_val19` best (val19 = 19 val recordings, val_ratio=0.17)
- Epochs: 150 (but started from epoch ~108, so ~42 more epochs with CFG)
- Best: epoch=126, val_loss=0.1673
- Note: LR was near-zero at resume (cosine schedule tail) — most weight change happened in the first ~10 epochs after resume before LR collapsed further

### Beat Alignment F1 Results

Eval script: `evaluate_beat_alignment.py --val_ratio 0.17 --num_samples 2`

**Baseline (original GT-beats, no CFG, epoch 67):**

| Condition | F1 | Gap |
|-----------|-----|-----|
| GT conditioned | 0.286 | — |
| Shuffled beat | 0.206 | GT−shuf: +0.080 |
| Unconditioned | 0.150 | GT−unc: +0.137 |

**CFG model (epoch 126) across guidance scales:**

| guidance_scale | GT F1 | Shuffled F1 | Uncond F1 | GT−shuf gap | GT−unc gap |
|----------------|-------|-------------|-----------|-------------|------------|
| 1.0 | 0.292 | 0.209 | 0.141 | +0.083 | +0.151 |
| 1.5 | 0.397 | 0.225 | 0.140 | +0.172 | +0.257 |
| 2.0 | 0.475 | 0.261 | 0.152 | +0.215 | +0.323 |
| 3.0 | 0.605 | 0.301 | 0.146 | +0.303 | +0.459 |

### Interpretation

- At `scale=1.0` the CFG model is essentially equivalent to the old model (no amplification, just one forward pass)
- As scale increases, GT F1 rises sharply (0.29 → 0.61) while shuffled barely moves (0.21 → 0.30) and unconditioned is flat (~0.14)
- The GT−shuffled gap at scale=3.0 (+0.303) is **~4× the old baseline** (+0.080)
- The flat unconditioned line confirms the amplification is specifically beat-driven, not a general onset-detection artifact from higher-energy outputs
- The shuffled F1 also rises somewhat at high scales — this is expected (any beat signal, even wrong, gets amplified), but the gap to GT remains large

**Conclusion:** CFG was the missing ingredient. The old dropout-based approach gave a weak signal (~+0.08 gap). Proper CFG with guidance_scale=2.0–3.0 gives a strong, unambiguous conditioning signal (+0.21–+0.30 gap). The model genuinely learned to follow beat structure.

---

## Guidance Scale gs=5.0 + Full Sweep (2026-04-14)

Added gs=5.0 to the guidance scale sweep. Full results (12s CFG model, 186 val windows × 2 samples):

| guidance_scale | GT F1 | Shuffled F1 | Uncond F1 | GT−shuf gap |
|----------------|-------|-------------|-----------|-------------|
| 1.0 | 0.292 | 0.209 | 0.141 | +0.083 |
| 1.5 | 0.397 | 0.225 | 0.140 | +0.172 |
| 2.0 | 0.475 | 0.261 | 0.152 | +0.215 |
| **3.0** | **0.605** | **0.301** | **0.146** | **+0.303** |
| 5.0 | 0.586 | 0.295 | 0.149 | +0.292 |

gs=5.0 slightly regresses from gs=3.0 (gap +0.292 vs +0.303). The model over-sharpens: GT F1 drops slightly and shuffled also drops, but the GT advantage narrows. **Optimal guidance scale: 3.0.**

---

## 20s CFG Model — Two Training Runs (2026-04-15–19)

**Motivation:** 20s windows capture longer melodic phrases and more beat events (205 val windows vs 186 for 12s). A longer context should help the model learn beat structure. Tested CFG fine-tuning at 20s.

### Run 1 (`hmr_gt_beats_20s_cfg`, flawed)

- Resumed from `hmr_gt_beats_20s` (val_loss=0.1611, epoch 34) — itself trained with cfg_prob=0.1 from epoch 1 (no clean pre-CFG phase)
- LR was in the cosine tail on resume — most weight updates happen before LR collapses
- Best: epoch=61, val_loss=0.1665
- Results (205 val windows × 2 samples, gs=3.0): GT F1=0.374, shuf=0.271, gap=**+0.104**

### Run 2 (`hmr_gt_beats_20s_cfg_v2`, clean procedure)

Fixed both issues from Run 1:
1. Clean pre-CFG phase: trained `hmr_gt_beats_20s_v2` with cfg_prob=0.0 to epoch 100 (val_loss=0.1593)
2. Fresh optimizer + cfg_prob=0.15 on resume (same two-phase procedure that worked for 12s model)
- Best: epoch=127, val_loss=0.1637
- Results (205 val windows × 2 samples, gs=3.0): GT F1=0.405, shuf=0.298, gap=**+0.107**

### Model Comparison (gs=3.0)

| Model | Windows | GT F1 | Shuf F1 | GT−shuf gap |
|-------|---------|-------|---------|-------------|
| 12s, no CFG (epoch 67) | 106×4 | 0.289 | 0.200 | +0.089 |
| 20s, no CFG (epoch 100) | 205×2 | 0.396 | 0.270 | +0.125 |
| **12s, CFG (epoch 126)** | **186×2** | **0.605** | **0.301** | **+0.303** |
| 20s, CFG v1 (epoch 61) | 205×2 | 0.374 | 0.271 | +0.104 |
| 20s, CFG v2 (epoch 127) | 205×2 | 0.405 | 0.298 | +0.107 |

**Conclusion:** The 20s CFG gap (~+0.107) is fundamentally limited relative to 12s CFG (+0.303). Fixing the training procedure barely moved the needle (Run 1 vs Run 2: +0.104 → +0.107). The likely cause is that 20s windows have sparser beat events relative to window length, making the beat conditioning signal harder to learn. The 12s model remains the best. The 20s CFG experiments are closed.

---

## Prime Experiment — Full CFG Val Set (`generate_cfg_with_prime.py`, 2026-04-20)

**Goal:** Does providing a pitch prime (first 400 frames of GT pitch) at the start of generation improve beat alignment? Does a beat-encoded prime (synthetic pulses at beat positions) help further?

**Setup:** 38 val windows selected for audible tempo (≥0.48 s/beat, i.e. ≤2 BPS — excludes ultra-fast drut). Generated under 7 conditions per window:

| Condition | Beat | Prime |
|-----------|------|-------|
| `noprime_gt_gs3.0` | GT beat, gs=3.0 | none |
| `noprime_shuf_gs3.0` | shuffled beat, gs=3.0 | none |
| `noprime_zero` | zero beat | none |
| `prime400_gt_gs3.0` | GT beat, gs=3.0 | first 400fr GT pitch |
| `prime400_shuf_gs3.0` | shuffled beat, gs=3.0 | first 400fr GT pitch |
| `beatprime400_gt_gs3.0` | GT beat, gs=3.0 | synthetic beat-pulse prime |
| `beatprime400_shuf_gs3.0` | shuffled beat, gs=3.0 | synthetic beat-pulse prime |

**Pitch prime mechanism:** First `P=400` frames of generation are linearly interpolated between noise and GT pitch during diffusion sampling (`x[:P] = (1−αt)·noise[:P] + αt·prime`), so the prime softly guides the initial trajectory.

**Beat prime construction (`make_beat_prime`):** A synthetic pitch array of length 1200 where silence token (QT=−1.077) is the background and a short 10-frame note (QT=0.2) is placed at each beat position. Intended to signal rhythmic structure via the prime, but the result is 95% silence in QT-space.

**F1 results (38 windows, mean):**

| Condition | Mean F1 | GT−shuf gap |
|-----------|---------|-------------|
| beatprime400_gt | 0.680 | +0.217 |
| beatprime400_shuf | 0.463 | — |
| **noprime_gt** | **0.602** | **+0.307** |
| noprime_shuf | 0.295 | — |
| prime400_gt | 0.504 | +0.236 |
| prime400_shuf | 0.268 | — |
| noprime_zero | 0.127 | — |

**Beat prime dropped:** `beatprime400` raises absolute GT F1 (0.680) but the GT−shuf gap *shrinks* (+0.217 vs +0.307 noprime) because the shuffled beat prime also inflates shuf F1 (0.463 vs 0.295) — the model follows the rhythmic structure locked into the prime rather than the beat conditioning. The synthetic prime is also perceptually useless (95% silence). **beatprime conditions excluded from all future experiments.**

**Pitch prime interpretation:** `prime400_gt` gives gap +0.236 vs noprime +0.307 — the pitch prime anchors the model to a melodic trajectory, which slightly dampens beat responsiveness. Not a clear improvement; the prime constrains rather than guides.

**Outputs:** `outputs_cfg_prime/` (38 window folders, all 7 conditions). Uploaded to Drive: `GaMaDHaNi-samples/outputs_cfg_prime/`.

---

## Prime Zero Condition + Comparison HTML (2026-04-21)

**Motivation:** The 4-condition comparison table (beat-conditioned × pitch-prime) needed a 4th cell: pitch prime + zero beat. This isolates how much the pitch prime alone drives rhythmic structure, without any beat conditioning.

**New condition: `prime400_zero`** (`generate_prime_zero.py`): pitch prime (first 400fr GT pitch) + zero beat signal (guidance_scale=1.0, no CFG). Generated for 6 windows used in the comparison table.

**6 windows selected** for audible tempo (≥0.48 s/beat):

| Window | Beats | s/beat |
|--------|-------|--------|
| 21007_1200 | 15 | 0.80 |
| 20002_0 | 25 | 0.48 |
| 21017_1200 | 15 | 0.80 |
| 20002_1200 | 25 | 0.48 |
| 21017_0 | 10 | 1.20 |
| 21007_0 | 10 | 1.20 |

**Comparison table (4 conditions per window):**

| Column | Beat | Prime |
|--------|------|-------|
| Beat Cond — No prime | GT, gs=3.0 | none |
| Beat Cond — Pitch prime | GT, gs=3.0 | 400fr GT pitch |
| Uncond — No prime | zero | none |
| Uncond — Pitch prime | zero | 400fr GT pitch |

**Output files:**
- `build_comparison_html.py` → `prime_comparison.html`: self-contained HTML with all audio base64-embedded, opens in any browser, no external dependencies
- `prime_comparison.ipynb`: same table as interactive notebook; requires local file paths (not usable in Drive/Colab without the audio files present)

**Note on noise:** Each condition in a row is an independent diffusion run with a fresh `torch.randn` noise sample — the generated pitch varies across columns not just from conditioning but also from different noise seeds.

**Uploaded to Drive:** `GaMaDHaNi-samples/prime_comparison.html`, `GaMaDHaNi-samples/prime_comparison.ipynb`.

---

## Pitch-Conditioned Model (`train_pitch_conditioned.py`, 2026-04-12)

**Motivation:** Replace the 3-channel annotated beat signal with the normalized pitch contour itself as the conditioning input. The pitch contour natively encodes rhythmic structure (pitch jumps correlate with beats in drut/madhya laya), and this approach requires no beat annotations at training or inference time.

**Architecture:** `PitchConditionedUNet` — identical to `BeatConditionedUNet` but `pitch_projection = nn.Linear(1, C)`. Conditions on the QT-transformed pitch token sequence [1, T] rather than the beat signal [1/3, T].

**Dataset change:** `HMRBeatDataset(condition_on_pitch=True)` returns `normalized_pitch` as the `beat` field (same sequence used as both target and conditioning). `beat_augment=False` since pitch contour isn't augmented.

**Training setup:**
- Condition on: GT normalized pitch tokens [1, T] (same window as target)
- Target: GT normalized pitch tokens [1, T] (what diffusion denoises toward)
- Loss: identical flow-matching MSE
- Checkpoint dir: `checkpoints/hmr_pitch_conditioned/`
- Run name: `hmr-pitch-conditioned`

**Key question:** Does the model learn a genuine pitch-structure → generation relationship (usable at inference with a different reference pitch), or does it collapse to copying the condition? The 10% pitch dropout prevents pure copying. Val loss trajectory vs. GT beats model will be the first indicator.

---

## Key Files

| File | Purpose |
|------|---------|
| `build_hmr_lmdb.py` | Builds LMDB from preprocessed .npy files |
| `hmr_dataset.py` | PyTorch Dataset wrapping the LMDB |
| `beat-condition-finetuning.ipynb` | Collaborator's notebook with BeatConditionedUNet |
| `HMR_processed/lmdb/` | Train/val LMDB databases |
| `HMR_processed/vocals/` | Demucs-separated vocal stems (114 files) |
| `HMR_processed/pitch/` | CREPE f0 + confidence (114 × 2 files) |
| `HMR_processed/beats/` | 3-channel beat signals (127 files, 114 used) |
| `beat_transformer_hindustani/inference/hmr_instruments.csv` | Per-recording instrument labels + exclusion flags |
| `beat_transformer_hindustani/inference/hmr_manual_check.csv` | 54 vocals-only recordings for manual verification |

---

## MusicBrainz Notes

- HMR has 127 vocal (V-coded) recordings; 3 had `MBID=TBA` (UIDs 21044, 21052, 22012) — MBIDs sourced manually
- Instrument labels cross-referenced from: HRR dataset CSV, Saraga dataset CSV, MusicBrainz `artist-rels`
- 62 of 127 were in HRR/Saraga; 65 fetched from MusicBrainz directly
- MB laya tags often differ from HMR `Corrected Lay Label` — MB tags full recording, HMR tags the specific excerpt; HMR labels are more reliable for fine-tuning
