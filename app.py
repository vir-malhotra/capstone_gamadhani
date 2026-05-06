"""
Gradio demo for Beat-Conditioned Hindustani Vocal Generation.

Two tabs:
  • Quick Listen  — instantly play 5 pre-generated examples × 3 beat conditions
  • Mix & Match   — pick any beat source + recording, generate live (needs GPU)

Run:
    python app.py
"""

import copy
import os
import sys

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs_cfg", "gs3")
HMR_VOCALS_DIR = "/home/vm2426/HMR_processed/vocals"
CKPT_PATH   = os.path.join(BASE_DIR, "checkpoints", "hmr_gt_beats_cfg", "best.ckpt")
HF_SNAPSHOT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi"
    "/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95"
)
PITCH_PATH  = os.path.join(HF_SNAPSHOT, "diffusion_pitch_model-model.ckpt")
QT_PATH     = os.path.join(HF_SNAPSHOT, "diffusion_pitch_model-qt.joblib")
AUDIO_PATH  = os.path.join(HF_SNAPSHOT, "pitch_to_audio_model-model.ckpt")
AUDIO_QT    = os.path.join(HF_SNAPSHOT, "pitch_to_audio_model-qt.joblib")
PITCH_CFG   = os.path.join(BASE_DIR, "configs", "diffusion_pitch_config.gin")
AUDIO_CFG   = os.path.join(BASE_DIR, "configs", "pitch_to_audio_config.gin")

SAMPLE_RATE   = 16000
NUM_STEPS     = 100
AUDIO_SEQ_LEN = 750
SINGER_ID     = 3

# ── 5 best pre-generated examples (ranked by GT−shuf gap) ────────────────────

EXAMPLES = {
    "Example 1  (UID 20002, gap +0.57)": "20002_0",
    "Example 2  (UID 23008, gap +0.51)": "23008_0",
    "Example 3  (UID 21042, gap +0.47)": "21042_0",
    "Example 4  (UID 21033, gap +0.40)": "21033_0",
    "Example 5  (UID 22004, gap +0.39)": "22004_0",
}
EXAMPLE_NAMES  = list(EXAMPLES.keys())
EXAMPLE_UIDS   = list(EXAMPLES.values())

CONDITIONS = {
    "GT beat  (this recording's taal)":           "gt_gs3.0",
    "Cross beat  (different recording's taal)":   "shuf_gs3.0",
    "No beat  (unconditioned)":                   "zero",
}
CONDITION_NAMES = list(CONDITIONS.keys())

SUBDIVISIONS = {
    "Beats only":    1,
    "Half-beats":    2,
    "Quarter-beats": 4,
}

# ── Click track helpers ───────────────────────────────────────────────────────

CLICK_FREQ     = 1000.0
CLICK_DURATION = 0.020
CLICK_GAIN     = 2.5
SUBCLICK_GAIN  = 1.5
SUBCLICK_FREQ  = 600.0


def _make_click(freq, duration, sr=SAMPLE_RATE):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t / (duration * 0.3))).astype(np.float32)


def _place_clicks(audio: np.ndarray, beat_np: np.ndarray,
                  peak: float, subdivisions: int, sr: int):
    spf = sr // 100
    beat_frames, _ = find_peaks(beat_np, height=0.5, distance=5)
    click = _make_click(CLICK_FREQ,    CLICK_DURATION,       sr)
    sub   = _make_click(SUBCLICK_FREQ, CLICK_DURATION * 0.5, sr)

    def _put(pos, tmpl, gain):
        end = min(pos + len(tmpl), len(audio))
        n = end - pos
        if n > 0:
            audio[pos:end] += tmpl[:n] * gain * peak

    for i, bf in enumerate(beat_frames):
        _put(int(bf * spf), click, CLICK_GAIN)
        if subdivisions > 1 and i + 1 < len(beat_frames):
            gap = beat_frames[i + 1] - bf
            for s in range(1, subdivisions):
                _put(int((bf + gap * s / subdivisions) * spf), sub, SUBCLICK_GAIN)


def overlay_clicks(wav_np: np.ndarray, beat_np: np.ndarray,
                   subdivisions: int = 1, sr: int = SAMPLE_RATE) -> np.ndarray:
    audio = wav_np.copy().astype(np.float64)
    peak  = np.abs(audio).max() + 1e-8
    _place_clicks(audio, beat_np, peak, subdivisions, sr)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def beat_as_clicks(beat_np: np.ndarray,
                   subdivisions: int = 1, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a silent track with only click sounds at beat positions."""
    n_samples = int(len(beat_np) / 100 * sr)
    audio = np.zeros(n_samples, dtype=np.float64)
    _place_clicks(audio, beat_np, peak=0.8, subdivisions=subdivisions, sr=sr)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ── Quick-listen helpers ──────────────────────────────────────────────────────

def quick_listen(example_name, condition_name, add_click, subdiv_name):
    folder   = EXAMPLES[example_name]
    cond_key = CONDITIONS[condition_name]
    folder_path = os.path.join(OUTPUTS_DIR, folder)

    wav, _ = torchaudio.load(os.path.join(folder_path, f"audio_{cond_key}.wav"))
    wav_np = wav.squeeze(0).numpy()

    if add_click:
        beat_np = np.load(os.path.join(folder_path, "beat.npy"))
        wav_np  = overlay_clicks(wav_np, beat_np, SUBDIVISIONS[subdiv_name])

    plot_path = os.path.join(folder_path, "plot_gs3.0.png")
    return (SAMPLE_RATE, wav_np), plot_path


def hear_beat_ql(example_name, subdiv_name):
    """Play the beat signal of the selected recording as pure clicks."""
    beat_np = np.load(os.path.join(OUTPUTS_DIR, EXAMPLES[example_name], "beat.npy"))
    return (SAMPLE_RATE, beat_as_clicks(beat_np, SUBDIVISIONS[subdiv_name]))


def hear_beat_mm(beat_example, subdiv_name):
    beat_np = np.load(os.path.join(OUTPUTS_DIR, EXAMPLES[beat_example], "beat.npy"))
    return (SAMPLE_RATE, beat_as_clicks(beat_np, SUBDIVISIONS[subdiv_name]))


def preview_window(example_name):
    """Play the pre-generated GT-beat audio for a window."""
    folder = os.path.join(OUTPUTS_DIR, EXAMPLES[example_name])
    wav, _ = torchaudio.load(os.path.join(folder, "audio_gt_gs3.0.wav"))
    return (SAMPLE_RATE, wav.squeeze(0).numpy())


def hear_gt_vocal(example_name):
    """Play the 12s GT vocal window with the most sustained pitch activity."""
    uid  = EXAMPLES[example_name].split("_")[0]
    path = os.path.join(HMR_VOCALS_DIR, f"{uid}_vocals.wav")
    wav, sr = torchaudio.load(path)
    wav = wav.mean(dim=0)  # mono

    # Use pitch confidence to find the best 12s window
    conf_path = f"/home/vm2426/HMR_processed/pitch/{uid}_confidence.npy"
    if os.path.exists(conf_path):
        conf = np.load(conf_path)
        window_frames = 1200  # 12s at 100Hz
        best_start = 0
        best_score = -1.0
        for i in range(0, len(conf) - window_frames, 100):
            score = float((conf[i:i + window_frames] > 0.5).mean())
            if score > best_score:
                best_score, best_start = score, i
        start_sample = int(best_start / 100 * sr)
    else:
        start_sample = 0

    clip = wav[start_sample: start_sample + sr * 12]
    return (sr, clip.numpy())


# ── BeatConditionedUNet (matches checkpoint) ──────────────────────────────────

class BeatConditionedUNet(nn.Module):
    def __init__(self, pretrained_unet):
        super().__init__()
        self.unet = copy.deepcopy(pretrained_unet)
        self.beat_projection = nn.Linear(1, self.unet.initial_projection.out_channels)
        self.unet.inp_dim = 1

    @property
    def device(self): return next(self.parameters()).device

    def forward(self, x, time, beat):
        x = self.unet.initial_projection(x)
        if beat.ndim == 3: beat = beat.transpose(1, 2)
        elif beat.ndim == 2: beat = beat.unsqueeze(-1)
        x = x + self.beat_projection(beat).transpose(1, 2)
        time = self.unet.positional_encoding(time)
        cat  = lambda a, t: torch.cat([a, t.unsqueeze(2).expand(-1,-1,a.shape[-1])], dim=-2)
        skips = []
        for dl in self.unet.downsample_layers:
            skips.append(x); x = cat(x, time); x = dl(x)
        skips.append(x)
        x = x.permute(0,2,1); x = self.unet.attention_layers(x); x = x.permute(0,2,1)
        for ul in self.unet.upsample_layers:
            x = cat(x, time); x = torch.cat([x, skips.pop(-1)], dim=1); x = ul(x)
        x = torch.cat([x, skips.pop(-1)], dim=1)
        return self.unet.final_projection(x)

    def sample(self, beat: torch.Tensor, guidance_scale: float = 3.0) -> torch.Tensor:
        b    = beat.shape[0]
        T    = beat.shape[-1]
        noise = torch.randn(b, self.unet.inp_dim, T, device=self.device)
        pn, pad  = self.unet.pad_to(noise, self.unet.strides_prod)
        pb, _    = self.unet.pad_to(beat.to(self.device), self.unet.strides_prod)
        null     = torch.zeros_like(pb)
        t_arr    = torch.ones(b, device=self.device)
        with torch.no_grad():
            for t in np.linspace(0, 1, NUM_STEPS + 1)[:-1]:
                tt = torch.tensor(t, device=self.device)
                pred_c = self.forward(pn, tt * t_arr, pb)
                pred_n = self.forward(pn, tt * t_arr, null)
                pred   = pred_n + guidance_scale * (pred_c - pred_n)
                pn     = pn + (1.0 / NUM_STEPS) * pred
        return self.unet.unpad(pn, pad)


# ── Lazy model loading ────────────────────────────────────────────────────────

_models = {}

def _load_models():
    if _models:
        return _models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from gamadhani.utils.generate_utils import load_pitch_fns, load_audio_fns
    import gamadhani.utils.pitch_to_audio_utils as p2a

    pitch_model, _, pitch_task_fn, invert_pitch_fn, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=PITCH_CFG, device=device)

    beat_model = BeatConditionedUNet(pitch_model).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    beat_model.load_state_dict(ckpt["state_dict"])
    beat_model.eval()

    audio_model, audio_qt, _, invert_audio_fn = load_audio_fns(
        audio_path=AUDIO_PATH, qt_path=AUDIO_QT, config_path=AUDIO_CFG, device=device)
    audio_model.eval()

    qt = joblib.load(QT_PATH)

    _models.update(dict(
        beat_model=beat_model, audio_model=audio_model,
        invert_pitch_fn=invert_pitch_fn, invert_audio_fn=invert_audio_fn,
        qt=qt, p2a=p2a, device=device,
    ))
    return _models


# ── Audio synthesis from QT-space pitch ──────────────────────────────────────

def _pitch_to_audio(pitch_qt, models):
    qt, audio_model, invert_audio_fn, p2a, device = (
        models["qt"], models["audio_model"], models["invert_audio_fn"],
        models["p2a"], models["device"],
    )
    tokens = qt.inverse_transform(pitch_qt.reshape(-1, 1))
    tokens = torch.tensor(tokens, dtype=torch.float32).reshape(1, 1, -1)
    tokens = torch.round(tokens)
    tokens[tokens < 200] = float("nan")
    interp = p2a.interpolate_pitch(tokens, AUDIO_SEQ_LEN).to(device)
    interp = torch.nan_to_num(interp, nan=196.0).squeeze(1).float()
    singer = torch.tensor([SINGER_ID], device=device)
    noise  = torch.randn(1, audio_model.inp_dim, AUDIO_SEQ_LEN, device=device)
    pn, pad = audio_model.pad_to(noise, audio_model.strides_prod)
    pf, _   = audio_model.pad_to(interp, audio_model.strides_prod)
    t_arr   = torch.ones(1, device=device)
    with torch.no_grad():
        for t in np.linspace(0, 1, NUM_STEPS + 1)[:-1]:
            tt = torch.tensor(t)
            u  = audio_model.forward(pn, tt*t_arr, pf, singer, drop_tokens=False, drop_all=True)
            c  = audio_model.forward(pn, tt*t_arr, pf, singer, drop_tokens=False, drop_all=False)
            pn = pn + (1.0/NUM_STEPS) * (3.0*c + (1-3.0)*u)
        wav = audio_model.unpad(pn, pad)
        wav = invert_audio_fn(wav)
    return wav[0].unsqueeze(0).cpu()


# ── Mix-and-match pitch plot ──────────────────────────────────────────────────

def _make_plot(beat_np, pitch_hz):
    t = np.arange(len(beat_np)) / 100.0
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 5), gridspec_kw={"hspace": 0.4})
    ax0.plot(t, beat_np, color="black", linewidth=0.8)
    ax0.set_title("Beat signal"); ax0.set_ylabel("Amplitude"); ax0.set_xlim(t[0], t[-1])
    voiced = pitch_hz > 0
    ax1.plot(t[voiced], pitch_hz[voiced], ".", markersize=1.5, color="steelblue")
    for bp in np.where(beat_np > 0.5)[0]:
        ax1.axvline(bp / 100.0, color="red", alpha=0.15, linewidth=0.5)
    ax1.set_title("Generated pitch contour"); ax1.set_ylabel("f0 (Hz)"); ax1.set_xlim(t[0], t[-1])
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


# ── Live mix-and-match generation ────────────────────────────────────────────

def live_generate(beat_example, melody_example, guidance_scale, add_click, subdiv_name):
    try:
        models = _load_models()
    except Exception as e:
        return None, None, f"Model loading failed: {e}"

    beat_folder   = os.path.join(OUTPUTS_DIR, EXAMPLES[beat_example])
    melody_folder = os.path.join(OUTPUTS_DIR, EXAMPLES[melody_example])

    beat_np = np.load(os.path.join(beat_folder, "beat.npy"))
    beat_t  = torch.tensor(beat_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    pitch_qt = models["beat_model"].sample(beat_t, guidance_scale=guidance_scale)
    pitch_qt = pitch_qt.squeeze().cpu().numpy()
    pitch_hz = models["invert_pitch_fn"](f0=pitch_qt).flatten()

    wav = _pitch_to_audio(pitch_qt, models)
    wav_np = wav.squeeze(0).numpy()

    if add_click:
        wav_np = overlay_clicks(wav_np, beat_np, SUBDIVISIONS[subdiv_name])

    plot_img = _make_plot(beat_np, pitch_hz)
    return (SAMPLE_RATE, wav_np), plot_img, "Done."


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Beat-Conditioned Hindustani Vocal Generation") as demo:
    gr.Markdown("""
# Beat-Conditioned Hindustani Vocal Generation
Fine-tuned [GaMaDHaNi](https://arxiv.org/abs/2408.12658) Stage 1 to generate pitch contours aligned with taal beat structure in Hindustani classical music.
""")

    with gr.Accordion("What is this? (click to expand)", open=False):
        gr.Markdown("""
## Background
**GaMaDHaNi** is a two-stage AI model for Hindustani vocal music generation. Stage 1 generates a pitch contour (the melody shape), and Stage 2 turns that into audio. This demo extends Stage 1 so it can be conditioned on a **taal beat signal** — making the generated melody rhythmically aware.

**Taal** is the rhythmic cycle in Hindustani classical music (e.g. Teentaal = 16 beats, Ektaal = 12 beats). The model is given a pulse signal marking each beat position and learns to place melodic phrases and note onsets in alignment with it.

---

## Glossary

**GT beat (ground truth beat)**
The actual annotated beat positions for that recording. Using this gives the model the correct taal structure to align to — this is the "ideal" conditioning.

**Shuffled / cross beat**
The beat signal taken from a *different* recording. The model still tries to align to the beat, but the taal structure no longer matches the melodic context. Used to test whether the model is genuinely responding to the beat signal or just generating freely.

**Unconditioned (no beat / zero beat)**
The beat signal is set to all zeros — the model generates freely with no rhythmic guidance. Gives a baseline for what the model produces without beat conditioning.

**Real vocalist**
The original vocal audio from the dataset (demucs-separated from the HMR recording). This is what a human singer actually sounds like — not generated.

**AI baseline (GT beat)**
A pre-generated output for that window using the recording's own beat. Useful as a reference point when comparing against a new mixed-beat generation.

**Generation window**
A 12-second excerpt from one of the validation recordings. The model generates a new pitch contour for this time slot, conditioned on the selected beat signal.

**Beat source**
Which recording's taal structure (beat pattern) to use as conditioning. In Mix & Match mode you can combine any beat source with any generation window.

**Guidance scale**
Controls how strongly the model follows the beat signal. Higher = more rhythmically rigid. Optimal value is 3.0 — above 5.0 the output starts to degrade.

**Click track**
Audible clicks overlaid at beat positions so you can hear where the beats fall. Subdivisions add additional softer clicks at half-beat or quarter-beat positions within each cycle.

**GT−shuf gap**
The key evaluation metric: how much better the model aligns to the *correct* beat versus a random one. A larger gap means the model is genuinely responding to the beat signal.
""")

    gr.Markdown("---")

    with gr.Tabs():

        # ── Tab 1: Quick Listen ───────────────────────────────────────────────
        with gr.Tab("Quick Listen"):
            gr.Markdown(
                "Play pre-generated examples instantly. "
                "Each example is a 12-second window from the validation set."
            )
            with gr.Row():
                ql_example   = gr.Dropdown(EXAMPLE_NAMES, value=EXAMPLE_NAMES[0],
                                           label="Recording")
                ql_condition = gr.Dropdown(CONDITION_NAMES, value=CONDITION_NAMES[0],
                                           label="Beat condition")
                ql_subdiv    = gr.Dropdown(list(SUBDIVISIONS.keys()),
                                           value="Beats only", label="Click subdivisions")
            with gr.Row():
                ql_beat_btn = gr.Button("Hear beat pattern")
                ql_gt_btn   = gr.Button("Hear GT vocal")
                ql_clicks   = gr.Checkbox(label="Add clicks to output", value=False)
                ql_play_btn = gr.Button("Play", variant="primary")
            with gr.Row():
                ql_beat_audio = gr.Audio(label="Beat pattern (clicks only)", type="numpy")
                ql_gt_audio   = gr.Audio(label="Real vocalist", type="numpy")
                ql_audio      = gr.Audio(label="Generated output", type="numpy")
            ql_plot = gr.Image(label="Pitch contour + beat overlay")

            ql_example.change(lambda: (None, None, None),
                              outputs=[ql_beat_audio, ql_gt_audio, ql_audio])
            ql_condition.change(lambda: None, outputs=[ql_audio])
            ql_beat_btn.click(hear_beat_ql,
                              inputs=[ql_example, ql_subdiv],
                              outputs=[ql_beat_audio])
            ql_gt_btn.click(hear_gt_vocal,
                            inputs=[ql_example],
                            outputs=[ql_gt_audio])
            ql_play_btn.click(quick_listen,
                              inputs=[ql_example, ql_condition, ql_clicks, ql_subdiv],
                              outputs=[ql_audio, ql_plot])

        # ── Tab 2: Mix & Match ────────────────────────────────────────────────
        with gr.Tab("Mix & Match  (Live Generation)"):
            gr.Markdown(
                "Pick any beat source and any generation window independently. "
                "Hear the taal pattern first, then generate. "
                "Requires a loaded model — generation takes ~20–40 seconds on GPU."
            )
            with gr.Row():
                mm_beat   = gr.Dropdown(EXAMPLE_NAMES, value=EXAMPLE_NAMES[0],
                                        label="Beat source  (taal structure to follow)")
                mm_melody = gr.Dropdown(EXAMPLE_NAMES, value=EXAMPLE_NAMES[1],
                                        label="Generation window  (val recording context)")
                mm_subdiv = gr.Dropdown(list(SUBDIVISIONS.keys()),
                                        value="Beats only", label="Click subdivisions")
            with gr.Row():
                mm_beat_btn    = gr.Button("Hear beat pattern")
                mm_preview_btn = gr.Button("AI baseline (GT beat)")
                mm_gt_btn      = gr.Button("Real vocalist")
                mm_clicks      = gr.Checkbox(label="Add clicks to output", value=True)
                mm_gs          = gr.Slider(1.0, 5.0, value=3.0, step=0.5,
                                           label="Guidance scale")
                mm_gen_btn     = gr.Button("Generate", variant="primary")
            with gr.Row():
                mm_beat_audio    = gr.Audio(label="Beat pattern (clicks only)", type="numpy")
                mm_preview_audio = gr.Audio(label="AI baseline (pre-generated, GT beat)", type="numpy")
                mm_gt_audio      = gr.Audio(label="Real vocalist", type="numpy")
                mm_audio         = gr.Audio(label="Generated audio", type="numpy")
            mm_plot   = gr.Image(label="Generated pitch contour + beat")
            mm_status = gr.Textbox(label="Status", interactive=False)

            mm_beat.change(lambda: None, outputs=[mm_beat_audio])
            mm_melody.change(lambda: (None, None, None),
                             outputs=[mm_preview_audio, mm_gt_audio, mm_audio])
            mm_beat_btn.click(hear_beat_mm,
                              inputs=[mm_beat, mm_subdiv],
                              outputs=[mm_beat_audio])
            mm_preview_btn.click(preview_window,
                                 inputs=[mm_melody],
                                 outputs=[mm_preview_audio])
            mm_gt_btn.click(hear_gt_vocal,
                            inputs=[mm_melody],
                            outputs=[mm_gt_audio])
            mm_gen_btn.click(live_generate,
                             inputs=[mm_beat, mm_melody, mm_gs, mm_clicks, mm_subdiv],
                             outputs=[mm_audio, mm_plot, mm_status])

if __name__ == "__main__":
    demo.launch(share=False)
