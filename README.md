# Beat-Conditioned GaMaDHaNi

Fine-tuning GaMaDHaNi's Stage 1 diffusion model to generate pitch contours rhythmically aligned to taal beat structure in Hindustani classical music.

**Best result:** GT−shuffled beat alignment F1 gap of **+0.303** (12s CFG model, guidance_scale=3.0), with statistically significant phase clustering (Rayleigh R=0.145, p<0.05) for drut/madhya laya.

Full experiment log: [`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md) | Full metrics reference: [`PROCESS.md`](PROCESS.md)

---

## Table of Contents

1. [Background](#background)
2. [Approach](#approach)
3. [Beat Transformer Experiments (Phase 0–1)](#beat-transformer-experiments-phase-01)
4. [Dataset Preparation](#dataset-preparation)
5. [Installation](#installation)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Audio Generation](#audio-generation)
9. [Experiment History & Results](#experiment-history--results)
10. [Key Findings](#key-findings)
11. [File Reference](#file-reference)

---

## Background

[GaMaDHaNi (Shikarpur et al., ISMIR 2024)](https://arxiv.org/abs/2408.12658) is a two-stage hierarchical model for Hindustani vocal music generation:
- **Stage 1:** Diffusion UNet that generates a pitch contour (f0 sequence at 100Hz, 12 seconds)
- **Stage 2:** Pitch-conditioned UNet that synthesizes audio from the pitch contour

This project fine-tunes Stage 1 to accept a **taal beat signal** as an additional conditioning input, following the [Sketch2Sound](https://arxiv.org/abs/2406.14185) approach. The goal: generate pitch contours whose melodic structure (note onsets, phrase boundaries) aligns with the rhythmic cycle of Hindustani classical music.

---

## Approach

### Conditioning Signal

A 1-channel beat pulse at 100Hz. For each annotated beat event position, a 5-frame rectangular kernel is placed and clipped to [0,1]. The HMR dataset provides ground-truth beat annotations from a Beat Transformer model.

### Architecture: `BeatConditionedUNet`

A single linear layer projects the beat signal into the UNet's feature space and adds it to the initial projection output:

```
beat [B, 1, T]
  → Linear(1, C)   # beat_projection
  → + UNet initial_projection(noisy_pitch)
  → rest of UNet (all weights fine-tuned)
```

All ~111M UNet parameters are fine-tuned (no frozen backbone), plus the one new Linear layer.

### Classifier-Free Guidance (CFG)

During training, with probability `cfg_prob=0.15`, the beat signal is zeroed for that batch sample — teaching the model to generate both with and without beat conditioning. At inference, two forward passes amplify the beat-specific component:

```
pred = pred_null + guidance_scale × (pred_cond − pred_null)
```

Optimal guidance_scale: **3.0**.

---

## Beat Transformer Experiments (Phase 0–1)

These scripts live in `beat_transformer_hindustani/inference/` and were run before any GaMaDHaNi fine-tuning. They require the Beat Transformer model weights in `beat_transformer_hindustani/inference/pretrained_models/` and the Beat Transformer code cloned to `/home/vm2426/beat_conditioning/Beat-Transformer/`.

### Run Beat Transformer inference on a single recording

```bash
cd beat_transformer_hindustani/inference
python run_beat_inference.py
# Runs on a single recording, saves beat/downbeat activations + decoded times
# Uses DBN post-processing via madmom
```

### Batch process all HMR recordings

```bash
cd beat_transformer_hindustani/inference
python process_hmr_batch.py
# Processes all HMR audio through Beat Transformer
# Saves raw activations + decoded beats/downbeats to HMR_processed/
```

### Evaluate Beat Transformer against HMR ground truth

```bash
cd beat_transformer_hindustani/inference
python evaluate_hmr.py
# Computes beat F1, sam precision/recall/F1 per recording
# Breaks down by laya (vilambit/madhya/drut) and taal
# Results saved in comprehensive_metrics.json
```

### Build ground-truth beat signals (3-channel 100Hz arrays)

```bash
cd beat_transformer_hindustani/inference
python build_beat_signals.py
# Reads .beats annotation files from HMR dataset
# Converts to 3-channel 100Hz numpy arrays (beat pulse, sam pulse, cycle position)
# Output: HMR_processed/beats/{uid}_beat.npy
```

### Fetch MusicBrainz instrument/laya/taal tags

```bash
cd beat_transformer_hindustani/inference
python fetch_mb_tags.py
# Queries MusicBrainz API for all 127 vocal HMR recordings
# Extracts: laya1, tala1, raga1, form1, instrument tags
# Output: hmr_mb_tags.csv

python merge_instruments.py
# Cross-references hmr_mb_tags.csv with HRR and Saraga CSVs
# Produces final instrument label per recording
# Output: hmr_instruments.csv
```

### Pitch extraction (vocal separation + CREPE)

```bash
cd beat_transformer_hindustani/inference
python extract_pitch_hmr.py
# Runs demucs htdemucs on all HMR vocals → HMR_processed/vocals/
# Runs CREPE at 100Hz on separated stems → HMR_processed/pitch/{uid}_f0.npy, {uid}_confidence.npy
```

### Add click tracks to audio for perceptual evaluation

```bash
cd beat_transformer_hindustani/inference
python add_clicks_peak_detection.py
# Adds clicks using peak detection on beat activations (no DBN)

python add_clicks_to_poorva.py
# Example: overlays GT beats + BT-predicted beats on Poorva recording
```

### Analyze by laya sections

```bash
cd beat_transformer_hindustani/inference
python analyze_by_laya_sections.py
# Segments each recording by laya label and computes metrics per section
# Output: laya_sections_analysis.json
```

---

## Dataset Preparation

### Source Data

**HMR (Hindustani Music Recordings):** 151 vocal recordings with CREPE f0 annotations and Beat Transformer beat annotations. Dataset available at [https://zenodo.org/records/1264742](https://zenodo.org/records/1264742).

**Filtering criteria:**
- Instrument code `V` (vocal lead) only
- No sarangi or violin accompaniment (cross-referenced via MusicBrainz)
- Manual exclusions: UIDs 20028, 20032, 21018, 21025, 22011, 23014, 23015, 23019

**Result:** 113 valid recordings.

**Train/val split:** `val_ratio=0.17, seed=42` → **94 train / 19 val** recordings. All checkpoints use this exact split.

### Step 1 — Vocal Separation + Pitch Extraction

```bash
cd beat_transformer_hindustani/inference
python extract_pitch_hmr.py
# Runs demucs htdemucs on all HMR vocals → HMR_processed/vocals/
# Runs CREPE at 100Hz on separated stems → HMR_processed/pitch/{uid}_f0.npy, {uid}_confidence.npy
```

### Step 3 — Beat Signal Construction

```bash
# Convert .beats annotation files → 3-channel 100Hz signal
# Channel 0: beat pulse (5-frame rectangular kernel, clipped [0,1])
# Channel 1: sam pulse (same, only at cycle start)
# Channel 2: continuous cycle position [0,1]
# Output: HMR_processed/beats/{uid}_beat.npy
python extract_hmr_beats.py
```

Only **channel 0** (beat pulse) is used as conditioning.

### Step 4 — Build LMDB

```bash
python build_hmr_lmdb.py
# Output: HMR_processed/lmdb/train, HMR_processed/lmdb/val
```

Each LMDB entry is an `AudioExample` protobuf with fields `pitch` [N], `beat` [3,N], `beat_confidence` [N].

### Windowing (done at dataset load time)

| Model | Window length | Stride | Val windows |
|-------|--------------|--------|-------------|
| 12s | 1200 frames | 1200 | 186 |
| 20s | 2000 frames | 1000 | 205 |

Pitch normalization pipeline (matches pretrained GaMaDHaNi):
- Hz → cents: `1200 × log2(f0 / 440)`
- Discretize + shift by +4915, bin by 10 → tokens [200–600]
- Silence token = 196 (token < 200)
- Apply QuantileTransformer from pretrained checkpoint

---

## Installation

Requires **Python 3.10**.

```bash
git clone <this-repo>
cd beat-conditioned-GaMaDHaNi
pip install -r requirements.txt
```

**Note:** `x-transformers` must be pinned to `1.30.2` (already in `requirements.txt`). Newer versions rename the LayerNorm weights and break checkpoint loading.

Additional packages required for specific workflows:

```bash
# Vocal separation (extract_pitch_hmr.py)
pip install demucs

# Pitch extraction (extract_pitch_hmr.py)
pip install crepe

# Beat tracking evaluation (evaluate_beat_tracker.py)
pip install madmom

# Evaluation scripts (evaluate_onset_cc.py, evaluate_phase.py, etc.)
pip install scipy
```

Pretrained GaMaDHaNi weights are downloaded automatically from HuggingFace (`kmaneeshad/GaMaDHaNi`) on first run. The paths used throughout:

```
PITCH_PATH  = ~/.cache/huggingface/.../diffusion_pitch_model-model.ckpt
QT_PATH     = ~/.cache/huggingface/.../diffusion_pitch_model-qt.joblib
AUDIO_PATH  = ~/.cache/huggingface/.../pitch_to_audio_model-model.ckpt
AUDIO_QT    = ~/.cache/huggingface/.../pitch_to_audio_model-qt.joblib
```

---

## Training

The best results come from a **two-phase procedure**. Do not skip Phase 1.

### Phase 1 — Pre-CFG (build beat conditioning without null examples)

```bash
python train_beat_conditioned.py \
    --checkpoint_dir checkpoints/my_model_phase1 \
    --seq_len 1200 \
    --batch_size 16 \
    --max_epochs 100 \
    --cfg_prob 0.0 \
    --run_name my-model-phase1
```

Train until val loss plateaus (~epoch 67–100, val loss ≈ 0.157). The model learns to follow the beat signal before it is ever asked to ignore it.

### Phase 2 — CFG fine-tune (teach the model to work without beats too)

```bash
python train_beat_conditioned.py \
    --checkpoint_dir checkpoints/my_model_cfg \
    --resume_from checkpoints/my_model_phase1/best.ckpt \
    --seq_len 1200 \
    --batch_size 16 \
    --max_epochs 150 \
    --cfg_prob 0.15 \
    --fresh_optimizer \
    --reset_best_val \
    --run_name my-model-cfg
```

`--fresh_optimizer` resets AdamW state and restarts the cosine LR schedule from lr=1e-4.  
`--reset_best_val` allows tracking a fresh best checkpoint from this run.  
Best checkpoint typically around epoch 126, val loss ≈ 0.167.

The two-phase approach is how the 12s model was trained.

### Training for 20s windows

Replace `--seq_len 1200 --batch_size 16` with `--seq_len 2000 --batch_size 8`. Two independent runs showed the 20s gap is fundamentally capped at ~+0.107 regardless of training procedure — the 12s model is superior.

### Existing checkpoints

The best model checkpoint is available on HuggingFace: [vir-malhotra/beat-conditioned-gamadhani](https://huggingface.co/vir-malhotra/beat-conditioned-gamadhani)

```bash
# Download best checkpoint
huggingface-cli download vir-malhotra/beat-conditioned-gamadhani best.ckpt --local-dir checkpoints/hmr_gt_beats_cfg/
```

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| `checkpoints/hmr_gt_beats/best.ckpt` | 0.1569 | 12s, no CFG, epoch 67 |
| `checkpoints/hmr_gt_beats_cfg/best.ckpt` | 0.1673 | **12s CFG — best model**, epoch 126 |
| `checkpoints/hmr_gt_beats_20s_v2/best.ckpt` | 0.1593 | 20s, no CFG, epoch 100 |
| `checkpoints/hmr_gt_beats_20s_cfg_v2/best.ckpt` | 0.1637 | 20s CFG, epoch 127 |

---

## Evaluation

### 1. Beat Alignment F1 (primary metric)

Generates pitch under three conditions (GT beat, shuffled beat from different recording, zero beat), detects note onsets (silence→voiced transitions + pitch jumps ≥ 1 semitone), computes F1 against GT beat positions (±50ms tolerance).

```bash
python evaluate_beat_alignment.py \
    --checkpoint checkpoints/hmr_gt_beats_cfg/best.ckpt \
    --guidance_scale 3.0 \
    --num_samples 2 \
    --val_ratio 0.17 \
    --output beat_alignment_cfg_gs3.json
```

For 20s model: add `--seq_len 2000 --window_stride 1000`.

**Key metric:** GT−shuffled gap. A large positive gap means the model responds specifically to the correct beat, not any beat-like signal.

### 2. Phase Distribution / Rayleigh R

Tests whether note onsets cluster at a consistent phase within the beat cycle — the statistically rigorous test for beat alignment.

```bash
python evaluate_phase.py \
    --checkpoint checkpoints/hmr_gt_beats_cfg/best.ckpt \
    --guidance_scale 3.0 \
    --exclude_vilambit
```

`--exclude_vilambit` is required: vocalists in slow tempo do not phase-lock to taal beats (validated in GT data), so including vilambit dilutes the signal without giving useful information.

### 3. Contour Cross-Correlation

Computes peak normalized cross-correlation between the generated pitch velocity signal (`|Δcents|` at 100Hz) and the GT beat pulse, within ±half-beat-period of lag=0.

```bash
python evaluate_contour_crosscorr.py \
    --checkpoint checkpoints/hmr_gt_beats_cfg/best.ckpt \
    --guidance_scale 3.0
```

### 4. GT Data Validation

Verifies that real vocal onsets actually correlate with beats in the ground truth (necessary sanity check before trusting model evals):

```bash
python evaluate_gt_vocal_beat_correlation.py
```

Result: GT gap +0.145 for drut laya, −0.039 for vilambit (confirms vilambit exclusion).

### 5. Cross-Correlation & IOI Consistency (early metric, pre-CFG models)

Computes peak cross-correlation between onset signal and beat signal, IOI consistency, and beat period accuracy for generated outputs in `outputs/`:

```bash
python evaluate_crosscorr.py
```

### 6. Onset Representation Comparison

Compares continuous |Δcents|, binary thresholds (25c–200c), and ISMIR 2021 §3 stable-note segmentation as onset representations for the CC metric:

```bash
python evaluate_onset_cc.py
# Runs on GT real vocal pitch + outputs_prime/ generated pitch
# Results printed to stdout; saved in pitch_values_crosscorr_results.json
```

### 7. Robustness Evaluation

Tests F1 under 13 beat signal degradation conditions (section dropout 10–50%, pulse dropout p=0.1–0.5, jitter ±2–20 frames):

```bash
python evaluate_robustness.py
# Evaluates both hmr_gt_beats and hmr_gt_beats_augmented checkpoints
# Results saved in robustness_results.json
```

### 8. Extracted Beats Model Training

To reproduce the extracted-beats baseline (Beat Transformer–extracted rather than GT annotations):

```bash
python train_beat_conditioned_extracted.py \
    --ckpt_dir checkpoints/hmr_extracted_beats \
    --epochs 100 \
    --batch_size 16 \
    --run_name hmr-extracted-beats
```

---

## Audio Generation

### Standard samples with click tracks

```bash
python generate_beat_samples.py \
    --checkpoint checkpoints/hmr_gt_beats_cfg/best.ckpt \
    --guidance_scale 3.0 \
    --out_dir outputs_cfg \
    --num_windows 38
```

Saves `audio_gt_click.wav`, `audio_shuf_click.wav`, `audio_zero_click.wav` per window (beat + 4-subdivision clicks overlaid). Stage 2 pitch-to-audio model is length-agnostic — the `sample_cfg` hardcoded `seq_len=750` is bypassed with a custom sampling loop.

Upload to Drive:
```bash
rclone copy outputs_cfg gdrive:GaMaDHaNi-samples/outputs_cfg --progress
```

### Early prime generation (non-CFG model, April 6)

```bash
python generate_with_prime.py
# Generates outputs_prime/ — pitch .npy files for noprime/pitch-prime/beat-prime conditions
# Used as input to synthesize_prime_audio.py
```

### Synthesize audio from prime outputs

```bash
python synthesize_prime_audio.py
# Converts outputs_prime/ pitch .npy → audio .wav via Stage 2
# Output: outputs_prime_audio/

python add_clicks_to_prime_audio.py
# Overlays GT beat click tracks on synthesized audio
# Click levels: sam (400Hz), beat (900Hz), half-beat (1400Hz), quarter-beat (2000Hz)
```

### Prime experiment (CFG model)

```bash
python generate_cfg_with_prime.py --out_dir outputs_cfg_prime --prime_len 400
```

Generates 5 conditions per window: `noprime_gt`, `noprime_shuf`, `noprime_zero`, `prime400_gt`, `prime400_shuf`. Beat-prime (`beatprime`) conditions are present in files but excluded from analysis (see Key Findings).

### Test Stage 2 at arbitrary audio lengths

```bash
python test_long_audio.py \
    --target_len 1500 \
    --seq_len 1200 \
    --out test_long_audio_out
# target_len in mel frames (62.5 fps): 750=12s, 1250=20s, 1500=24s
```

### Comparison table (self-contained HTML)

```bash
# Generate prime400_zero condition (pitch prime + zero beat) for 6 windows:
python generate_prime_zero.py

# Build self-contained HTML with base64-embedded audio:
python build_comparison_html.py
# → prime_comparison.html  (open in any browser, no external files needed)
```

---

## Experiment History & Results

### Phase 0 — Beat Transformer Inference on Hindustani Music (February 2026)

Before any GaMaDHaNi work, the project started with running a pre-trained **Beat Transformer** (trained on Western music) on Hindustani classical recordings to assess how well it detects beats and sams (cycle starts/downbeats).

**Setup:** Beat Transformer model (`beat_transformer_hindustani/`) run on vocal recordings. Demucs htdemucs used to separate stems before inference. DBN (Dynamic Bayesian Network) post-processing used for beat/downbeat tracking.

**Recordings tested:** Several individual raagas — Raag Sohani, Basanti Kedar, Poorva, and others. Both early (pre-tabla) and later (with tabla backing) sections tested, since the model's performance was expected to differ.

**Key findings:**
- **Basanti Kedar** achieved the best performance — cleanest beat/downbeat activations and highest F1. Reason: strong percussive energy from tabla + regular drut tempo gives the transformer clear rhythmic anchors.
- **Downbeat/sam detection was unreliable** across most recordings — DBN meter assumptions (defaulting to 4-beat) did not match actual taal cycle lengths (Teentaal=16, Ektaal=12, etc.)
- **Vilambit (slow tempo)** recordings had near-zero recall — the model was trained on Western music with much shorter beat periods; it could not track the long inter-beat intervals of vilambit.
- **Sam ≠ downbeat:** Sams (cycle starts) occur every 10–16 beats, not every 4. The DBN's fixed meter parameter had to be manually set per taal.
- **Later sections (with tabla) performed significantly better** than earlier sections (vocal alone), confirming percussive energy drives the detector.

**Metrics collected:** Precision/recall/F1 for beats and sams separately, across laya categories (vilambit/madhya/drut). Click tracks generated and overlaid on original audio for perceptual verification.

**Conclusion:** Beat Transformer can detect beats in drut/madhya Hindustani recordings with tabla accompaniment at moderate accuracy, but sam detection is unreliable without per-taal DBN configuration. This motivated using **ground-truth HMR beat annotations** for model training rather than extracted beats.

---

### Phase 1 — HMR Dataset Analysis (February 2026)

Received the **HMR (Hindustani Music Recordings)** dataset from collaborator — 151 vocal recordings with CREPE f0 annotations and Beat Transformer beat annotations (.beats files).

**Dataset stats explored:**
- Distribution across laya: heavy vilambit bias (~43%), drut and madhya underrepresented
- Distribution across taals: Teentaal and Ektaal dominant
- Annotation duration: all excerpts ~2 minutes (reason: HMR excerpts are fixed-length clips, not full recordings)

**Beat Transformer run on HMR:** Compared model-extracted beats against the HMR ground-truth annotations. Computed beat F1, recall, precision per recording and per laya category.

**Listening:** Generated click tracks combining ground-truth beats, Beat Transformer predictions, and original audio in a single folder per recording for perceptual comparison.

**Key finding:** Ground-truth annotations from HMR are far more reliable than Beat Transformer extraction, especially for vilambit and non-standard taals. This confirmed the training strategy: use GT annotations for Phase 1 of training, with extracted beats as a secondary comparison.

---

### Phase 2 — Dataset Filtering + MusicBrainz Instrument Labeling (February–March 2026)

**Problem:** HMR instrument labels (e.g. `V` = vocal, `R` = sarangi) are often inaccurate — recordings labeled "vocal only" frequently contain sarangi or violin, which overlap spectrally with the voice and would contaminate the pitch conditioning signal.

**Process:**
1. Cross-referenced HMR MBIDs against HRR and Saraga dataset CSVs (which have MusicBrainz-derived instrument labels)
2. For the 65 recordings not in those CSVs, fetched instrument tags directly from MusicBrainz API
3. Extracted "other tags" from MusicBrainz for each recording: `laya1`, `tala1`, `raga1`, `form1`
4. Manually flagged 54 recordings claiming "vocal only" for human verification
5. Removed recordings with confirmed sarangi or violin: UIDs **20028, 20032, 21018, 21025, 22011, 23014, 23015, 23019**

**Final dataset:** 113 valid recordings. Split: `val_ratio=0.17, seed=42` → 94 train / 19 val. This split is fixed across all checkpoints.

---

### Phase 3 — Architecture Decisions + Pipeline Setup (March 2026)

**Collaborator context:** Collaborator (original GaMaDHaNi author) shared a beat conditioning notebook with a toy sine wave dataset demonstrating that `BeatConditionedUNet` — a single linear projection layer added to the UNet's initial projection — works as a conditioning mechanism.

**Architecture discussions:**
- Compared approaches: ControlNet-style (frozen backbone + side network), cross-attention over beat frames, FiLM conditioning, input concatenation
- **Decision:** Sketch2Sound style — one `Linear(1, C)` beat projection added to the UNet's initial projection output, full fine-tune of all weights. Simplest approach consistent with collaborator's notebook.
- Started with **1-channel beat signal** (beat pulse only), with plan to extend to 3-channel (beat + sam + cycle position) — ultimately stayed at 1-channel throughout as it performed well.

**Data pipeline built:**
- Demucs `htdemucs` for vocal stem separation → `HMR_processed/vocals/`
- CREPE at 100Hz for pitch extraction → `HMR_processed/pitch/`
- Beat annotation → 3-channel 100Hz signal → `HMR_processed/beats/`
- LMDB built with protobuf `AudioExample` format (extended with `beat` and `beat_confidence` buffers)
- `HMRBeatDataset`: windowing (1200 frames = 12s), confidence filtering, pitch normalization matching pretrained GaMaDHaNi pipeline

**Bugs encountered and fixed:**
- `buffer.sample_rate` → `buffer.sampling_rate` (proto field name mismatch)
- Beat shape `[1, T]` → `[T]` (dataset returned wrong shape; training loop does `.unsqueeze(1)`)
- `x-transformers==2.0.0` → pinned to `1.30.2` (LayerNorm weight naming changed between versions, breaking checkpoint loading)

**Manual verification:** Beat/pitch alignment confirmed visually from LMDB — beat pulses align with f0 contour at expected positions for drut recordings.

---

### Phase 4 — Baseline Training (March 28)

First training run: 100 epochs, no CFG, GT beat annotations. Established that beat conditioning works at all.

| Condition | F1 | Gap |
|-----------|-----|-----|
| GT conditioned | 0.289 | — |
| Shuffled beat | 0.200 | +0.089 |
| Zero beat | 0.143 | — |

Per-window analysis revealed vilambit recordings (2–3 beats/window) contribute near-zero F1 regardless of condition — the onset-jump metric is inappropriate for slow tempo. Drut/madhya recordings show the strongest GT−shuf separation.

### Phase 5 — Beat Augmentation (March 30)

Added section dropout, tempo halving, and tempo doubling to the beat signal during training. Result: gap **dropped** from +0.089 to +0.035. Augmentation at p=0.3 was too aggressive — the model learned to generate without relying on the beat.

### Phase 6 — Extracted Beats (March 30)

Trained a parallel model conditioned on Beat Transformer–extracted beats (rather than GT annotations). Cross-correlation gap: **−0.0015** (essentially no signal). Phase analysis: Rayleigh R=0.054 (not significant). Extracted beats are too noisy to train effective conditioning.

### Phase 7 — Data Validation (March–April)

**GT vocal onset vs. beat correlation** (`evaluate_gt_vocal_beat_correlation.py`) across all 113 recordings, 1105 windows:

| Laya | N | GT F1 | Shuf F1 | Gap |
|------|---|-------|---------|-----|
| Drut | 358 | 0.270 | 0.125 | +0.145 |
| Madhya | 205 | 0.125 | 0.104 | +0.022 |
| Vilambit | 476 | 0.050 | 0.089 | **−0.039** |

Drut: hypothesis holds. Vilambit: vocalist ornaments freely between widely-spaced beats — GT actually underperforms shuffled. This validated that vilambit must be excluded from all beat alignment metrics.

**Cross-correlation & IOI consistency eval** (`evaluate_crosscorr.py`): three supplementary metrics on generated pitch — peak CC with beat signal, IOI consistency (std/beat_period), beat period accuracy.

| Model | CC gap | IOI ordering |
|-------|--------|-------------|
| GT beats model | +0.0105 | gt < shuf < zero ✓ |
| Extracted beats model | −0.0015 | flat ✗ |

Confirms GT beats are necessary for training.

### Phase 8 — Robustness Evaluation (March 30)

Both GT beats and augmented models tested under 13 beat signal degradation conditions (section dropout 10–50%, pulse dropout p=0.1–0.5, beat jitter ±2–20 frames). Key result: original model wins on every condition, and both degrade at the same rate. Augmentation provided no robustness benefit — it simply started from a weaker baseline. Results saved in `robustness_results.json`.

### Phase 9 — Phase Analysis (April)

Rayleigh R (circular statistics on onset phases within beat cycle), drut+madhya only:
- **GT beats model: R=0.145, significant (p<0.05)** — beat bin at 10.3% (>2× uniform)
- Extracted beats model: R=0.054, not significant

Drut sub-group: R=0.145; drut/madhya sub-group: R=0.162. These are the strongest statistical evidence that beat conditioning works.

### Phase 10 — Early Prime Eval (`generate_with_prime.py`, April 6)

First prime experiment — on the non-CFG GT beats model. All 116 val windows, 6 conditions: no prime, pitch prime at 100/200/400 frames, and beat-encoded prime (synthetic pitch with short pulses at beat positions) at 100/200/400 frames.

| Condition | Mean F1 | vs no-prime |
|-----------|---------|-------------|
| No prime, GT beat | 0.1399 | — |
| Pitch prime 400fr | **0.1504** | +0.011 |
| Beat-prime 400fr | 0.1351 | −0.005 |

Pitch prime helps marginally at 400fr; beat-encoded prime consistently hurts. Gains too small on a non-CFG model. This motivated the full prime experiment on the CFG model (Phase 12).

### Phase 11 — Full Contour Cross-Correlation (`evaluate_contour_crosscorr.py`, April 9–12)

Continuous pitch velocity (`|Δcents|` at 100Hz) cross-correlated with the GT beat pulse — a richer metric than binary onset F1.

**Bug found and fixed:** Initial results included the prime region (P=400 frames) in CC computation. For `beatprime400`, the prime encodes beat positions as note pulses → trivially inflated CC (0.2165 buggy vs. corrected 0.0390). Fixed by computing CC over `[P:]` only.

**Corrected results (116 windows):**

| Condition | Peak CC | GT−shuf gap |
|-----------|---------|-------------|
| GT real vocal | 0.0187 | +0.013 |
| noprime_gt generated | 0.0740 | — |
| prime400_shuf generated | 0.0115 | **+0.064** |
| beatprime400_gt generated | 0.0390 | +0.029 (was 0.217 buggy) |

### Phase 12 — Audio Synthesis + Click Tracks (April 7–9)

Converted pitch `.npy` files → audio via Stage 2 (`synthesize_prime_audio.py`). Click tracks overlaid at beat positions using a 4-level click hierarchy: sam (400Hz), beat (900Hz), half-beat (1400Hz), quarter-beat (2000Hz). Used to perceptually verify whether generated pitch transitions align with the beat.

Stage 2 is length-agnostic: `UNetPitchConditioned.forward()` accepts any length. Custom sampling loop in `generate_beat_samples.py` and `test_long_audio.py` bypasses `sample_cfg`'s hardcoded `seq_len=750`. Mel frame rate = 62.5 frames/s (750 frames = 12s, 1250 frames = 20s).

### Phase 13 — Architecture Analysis (April 11)

Identified why beat conditioning is structurally limited with a single linear layer:
1. `Linear(1, C)` at initial projection is the only entry point — signal must survive all downsampling/upsampling unchanged
2. No explicit beat loss — MSE on noise prediction can be minimised while ignoring the beat
3. Sparse beat signal — only a few frames per beat period carry signal

Considered but not implemented: multi-scale injection, input concatenation, FiLM conditioning, cross-attention. CFG turned out to be the higher-leverage change.

### Phase 14 — CFG Fix (April 13)

**Problem:** The original `BeatConditionedUNet` used `nn.Dropout(p=0.1)` on the projected beat embedding. This is regularization, not CFG — the model never sees a fully absent beat signal during training, so the two-pass CFG inference formula amplifies noise rather than the conditioning signal.

**Fix:** During training, zero the entire beat signal for `cfg_prob=0.15` of the batch. The model learns both conditional and unconditional generation as distinct states.

Result: Guidance scale sweep on 12s model (186 val windows × 2 samples):

| Guidance Scale | GT F1 | Shuf F1 | Gap |
|----------------|-------|---------|-----|
| 1.0 | 0.292 | 0.209 | +0.083 |
| 1.5 | 0.397 | 0.225 | +0.172 |
| 2.0 | 0.475 | 0.261 | +0.215 |
| **3.0** | **0.605** | **0.301** | **+0.303** |
| 5.0 | 0.586 | 0.295 | +0.292 |

CFG was the missing ingredient — gap improved from +0.089 to +0.303 (3.4×).

### Phase 15 — 20s Models (April 15–19)

Trained two 20s CFG models (Run 1: flawed LR resume + no clean pre-CFG phase; Run 2: clean two-phase procedure). Both converged to gap ~+0.107. 20s is fundamentally limited — sparser beat events relative to window length. 20s experiments closed.

### Phase 16 — Onset Representation Comparison (April 19)

Compared three onset representations for the CC metric (GT real vocal + generated pitch):

| Representation | GT−shuf gap (generated) |
|----------------|------------------------|
| Continuous \|Δcents\| | +0.035 |
| Binary 25c | +0.034 |
| Binary 50c | +0.026 |
| Stable-note (ISMIR 2021 §3) | +0.008 — and false positive on wrong-beat |

Continuous |Δcents| kept as default. ISMIR stable-note segmentation dropped (unreliable at 12s window length).

### Phase 17 — Prime Experiment (April 20)

38 val windows, 7 conditions. Key question: does a pitch prime (first 400 frames of GT pitch) improve beat alignment?

| Condition | Mean F1 | GT−shuf gap |
|-----------|---------|-------------|
| beatprime400_gt | 0.680 | +0.217 |
| beatprime400_shuf | 0.463 | — |
| **noprime_gt** | **0.602** | **+0.307** |
| noprime_shuf | 0.295 | — |
| prime400_gt | 0.504 | +0.236 |
| prime400_shuf | 0.268 | — |
| noprime_zero | 0.127 | — |

Beat prime dropped (95% silence in QT-space, inflates both GT and shuffled F1). Pitch prime slightly reduces the gap (+0.236 vs +0.307) — the prime anchors melodic trajectory and competes with beat conditioning.

### Phase 18 — Comparison Table (April 21)

Generated `prime400_zero` condition (pitch prime + zero beat) for 6 audible-tempo windows. Built `prime_comparison.html` — self-contained, base64-embedded audio, 4 conditions × 6 windows. Uploaded to Drive: `GaMaDHaNi-samples/prime_comparison.html`.

---

## Key Findings

> All findings below are scoped to the evaluation metrics used in this project (beat alignment F1, GT−shuffled gap, Rayleigh R, and peak cross-correlation). They may not generalise to other metrics or evaluation settings.

1. **CFG is essential.** Dropout ≠ CFG. Whole-signal zeroing during training + two-pass inference gives 3.4× better gap (+0.303 vs +0.089).

2. **Two-phase training matters.** Phase 1 (no CFG) builds strong beat conditioning; Phase 2 adds null examples. Skipping Phase 1 or resuming with a stale LR schedule produces much weaker results.

3. **12s beats 20s** (as measured by GT−shuf gap). Two independent 20s runs both cap at ~+0.107. Sparser beat events per window is the likely bottleneck.

4. **Guidance scale 3.0 is optimal** for this metric. gs=5 slightly over-sharpens.

5. **Vilambit excluded from phase analysis.** GT data shows vocalists in vilambit do not onset-align to beats under our F1 metric — negative gap in ground truth. Results for drut+madhya only.

6. **Beat augmentation reduced the gap** from +0.089 to +0.035 (section dropout p=0.3). May behave differently under other training or metric configurations.

7. **Extracted beats produced near-zero gap and non-significant Rayleigh R** in our experiments. GT annotations were required to get a meaningful training signal under these metrics.

8. **Pitch prime slightly reduced the GT−shuf gap** (+0.236 vs noprime +0.307) — the prime constrains melodic trajectory and appears to compete with beat conditioning under this metric.

9. **Beat prime is misleading** for this evaluation. 95% silence in QT-space inflates both GT and shuffled F1, making the gap unreliable as a measure of beat responsiveness.

10. **Continuous |Δcents| ≈ binary 25c for CC.** ISMIR 2021 stable-note segmentation showed false positives on the wrong-beat condition in our setup.

11. **Stage 2 is length-agnostic.** `UNetPitchConditioned.forward()` accepts arbitrary sequence lengths. Bypass `sample_cfg`'s hardcoded `seq_len=750` with a custom sampling loop. Mel rate = 62.5 frames/s.

---

## File Reference

### Scripts

| Script | Purpose |
|--------|---------|
| `build_hmr_lmdb.py` | Build LMDB from HMR preprocessed files (GT beats) |
| `build_hmr_lmdb_extracted.py` | Build LMDB variant conditioned on extracted (BT-predicted) beats |
| `extract_hmr_beats.py` | Convert .beats annotations → 3-channel 100Hz signal |
| `hmr_dataset.py` | `HMRBeatDataset` class — windowing, normalization, beat loading |
| `train_beat_conditioned.py` | Training (CFG, --seq_len, --fresh_optimizer, --reset_best_val) |
| `train_beat_conditioned_extracted.py` | Training variant using extracted beats LMDB |
| `evaluate_beat_alignment.py` | Beat alignment F1 evaluation |
| `evaluate_beat_alignment_extracted.py` | Beat alignment F1 for extracted-beats model |
| `evaluate_augmented.py` | Beat alignment F1 for beat-augmented model (3 conditions) |
| `evaluate_prime.py` | F1 eval on prime-conditioned outputs in outputs_prime/ (non-CFG model) |
| `evaluate_phase.py` | Rayleigh R phase analysis |
| `evaluate_contour_crosscorr.py` | Pitch velocity × beat CC metric |
| `evaluate_crosscorr.py` | Cross-correlation + IOI consistency on outputs/ (early pre-CFG metric) |
| `evaluate_onset_cc.py` | Onset representation comparison (continuous/binary/stable-note) |
| `evaluate_pitch_values_crosscorr.py` | Raw pitch values (not velocity) × beat CC — alternative CC formulation |
| `evaluate_gt_vocal_beat_correlation.py` | GT data validation: do real onsets align with beats? |
| `evaluate_robustness.py` | F1 under beat signal degradation (jitter, dropout, section masking) |
| `evaluate_beat_tracker.py` | Run madmom beat tracker on generated WAVs and score against GT |
| `generate_beat_samples.py` | Audio generation + click tracks (gt/shuf/zero) |
| `generate_with_prime.py` | Early prime generation: noprime/pitch-prime/beat-prime (non-CFG model) |
| `generate_cfg_with_prime.py` | Prime experiment generation for CFG model (noprime/prime400/beatprime) |
| `generate_prime_zero.py` | prime400_zero condition for 6 comparison windows |
| `synthesize_prime_audio.py` | Convert outputs_prime/ pitch .npy → audio .wav via Stage 2 |
| `add_clicks_to_prime_audio.py` | Overlay GT beat click tracks on synthesized prime audio |
| `test_long_audio.py` | Test Stage 2 at arbitrary audio lengths beyond 12s |
| `build_comparison_html.py` | Self-contained HTML comparison table with base64 audio |
| `train_pitch_conditioned.py` | (Unused) pitch-conditioned model — checkpoint dir is empty |

### Key Data Files

| File | Contents |
|------|---------|
| `checkpoints/hmr_gt_beats_cfg/best.ckpt` | Best model (12s CFG, epoch 126) |
| `outputs_cfg_prime/` | 38 val windows × 7 conditions (pitch .npy + audio .wav) |
| `prime_comparison.html` | Self-contained comparison table (Drive copy is authoritative) |
| `beat_alignment_cfg_gs3.0.json` | F1 eval results for best model |
| `phase_eval_novilambit_log.txt` | Rayleigh R results (drut+madhya) |
| `contour_crosscorr_results.json` | CC metric results |
| `pitch_values_crosscorr_results.json` | Onset representation comparison results |
| `robustness_results.json` | Robustness eval under beat degradation |
| `PROCESS.md` | Comprehensive results reference, glossary, all tables |
| `IMPLEMENTATION_NOTES.md` | Chronological experiment log with raw numbers |

### Google Drive ([GaMaDHaNi-samples/](https://drive.google.com/drive/folders/1CgBsSsK5L6gZi4_V8iSNj8x1WVuu_zER?usp=sharing))

| Folder/File | Contents |
|-------------|---------|
| `outputs_cfg_prime/` | 38 windows × 7 prime conditions |
| `prime_comparison.html` | Self-contained comparison table |
| `prime_comparison.ipynb` | Notebook version (requires local audio files) |

---

## Reference

Base model: [GaMaDHaNi — Shikarpur et al., ISMIR 2024](https://arxiv.org/abs/2408.12658)  
Conditioning approach: [Sketch2Sound — García-Forte et al., 2024](https://arxiv.org/abs/2406.14185)  
Beat annotations: Beat Transformer (HMR dataset)  
Onset representation comparison: [ISMIR 2021 §3](https://archives.ismir.net/ismir2021/paper/000082.pdf)
