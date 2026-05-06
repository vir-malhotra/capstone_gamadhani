"""
Gradio demo — Beat-Conditioned Hindustani Vocal Generation (Quick Listen)
Serves pre-generated examples; no GPU required.
"""

import os
import numpy as np
import gradio as gr
import soundfile as sf
from scipy.signal import find_peaks

SAMPLE_RATE  = 16000
EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

EXAMPLES = {
    "Example 1  (UID 20002, gap +0.57)": "20002_0",
    "Example 2  (UID 23008, gap +0.51)": "23008_0",
    "Example 3  (UID 21042, gap +0.47)": "21042_0",
    "Example 4  (UID 21033, gap +0.40)": "21033_0",
    "Example 5  (UID 22004, gap +0.39)": "22004_0",
}
EXAMPLE_NAMES = list(EXAMPLES.keys())

CONDITIONS = {
    "GT beat  (this recording's taal)":         "gt_gs3.0",
    "Cross beat  (different recording's taal)": "shuf_gs3.0",
    "No beat  (unconditioned)":                 "zero",
}
CONDITION_NAMES = list(CONDITIONS.keys())

SUBDIVISIONS = {"Beats only": 1, "Half-beats": 2, "Quarter-beats": 4}

CLICK_FREQ     = 1000.0
CLICK_DURATION = 0.020
CLICK_GAIN     = 2.5
SUBCLICK_GAIN  = 1.5
SUBCLICK_FREQ  = 600.0


def _make_click(freq, duration, sr=SAMPLE_RATE):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t / (duration * 0.3))).astype(np.float32)


def _place_clicks(audio, beat_np, peak, subdivisions, sr=SAMPLE_RATE):
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


def overlay_clicks(wav_np, beat_np, subdivisions=1):
    audio = wav_np.copy().astype(np.float64)
    _place_clicks(audio, beat_np, peak=np.abs(audio).max() + 1e-8,
                  subdivisions=subdivisions)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def beat_as_clicks(beat_np, subdivisions=1):
    audio = np.zeros(int(len(beat_np) / 100 * SAMPLE_RATE), dtype=np.float64)
    _place_clicks(audio, beat_np, peak=0.8, subdivisions=subdivisions)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def quick_listen(example_name, condition_name, add_click, subdiv_name):
    folder   = os.path.join(EXAMPLES_DIR, EXAMPLES[example_name])
    cond_key = CONDITIONS[condition_name]
    wav_np, sr = sf.read(os.path.join(folder, f"audio_{cond_key}.wav"))
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    wav_np = wav_np.astype(np.float32)
    if add_click:
        beat_np = np.load(os.path.join(folder, "beat.npy"))
        wav_np  = overlay_clicks(wav_np, beat_np, SUBDIVISIONS[subdiv_name])
    plot_path = os.path.join(folder, "plot_gs3.0.png")
    return (sr, wav_np), plot_path


def hear_beat(example_name, subdiv_name):
    beat_np = np.load(os.path.join(EXAMPLES_DIR, EXAMPLES[example_name], "beat.npy"))
    return (SAMPLE_RATE, beat_as_clicks(beat_np, SUBDIVISIONS[subdiv_name]))


def hear_gt_vocal(example_name):
    uid  = EXAMPLES[example_name].split("_")[0]
    path = os.path.join(EXAMPLES_DIR, f"{uid}_gt_vocal.wav")
    wav_np, sr = sf.read(path)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    return (sr, wav_np.astype(np.float32))


with gr.Blocks(title="Beat-Conditioned Hindustani Vocal Generation") as demo:
    gr.Markdown("""
# Beat-Conditioned Hindustani Vocal Generation
Fine-tuned [GaMaDHaNi](https://arxiv.org/abs/2408.12658) Stage 1 to generate pitch contours
aligned with taal beat structure in Hindustani classical music.
""")

    with gr.Accordion("What is this? (click to expand)", open=False):
        gr.Markdown("""
## Background
**GaMaDHaNi** is a two-stage AI model for Hindustani vocal music generation. Stage 1 generates
a pitch contour (the melody shape), and Stage 2 turns that into audio. This demo extends Stage 1
so it can be conditioned on a **taal beat signal** — making the generated melody rhythmically aware.

**Taal** is the rhythmic cycle in Hindustani classical music (e.g. Teentaal = 16 beats,
Ektaal = 12 beats). The model is given a pulse signal marking each beat position and learns to
place melodic phrases and note onsets in alignment with it.

---

## Glossary

**GT beat (ground truth beat)**
The actual annotated beat positions for that recording. Using this gives the model the correct
taal structure to align to — this is the "ideal" conditioning.

**Cross beat**
The beat signal taken from a *different* recording. The model still tries to align to beats,
but the taal structure no longer matches. Used to test whether the model genuinely responds to
the beat or just generates freely.

**No beat (unconditioned)**
The beat signal is all zeros — the model generates freely with no rhythmic guidance.

**Real vocalist**
The original vocal audio from the dataset (demucs-separated). This is what a human singer
actually sounds like — not generated.

**Guidance scale**
Controls how strongly the model follows the beat signal. Optimal value is 3.0.

**Click track**
Audible clicks overlaid at beat positions. Subdivisions add softer clicks at half-beat or
quarter-beat positions within each cycle.

**GT−shuf gap**
Key evaluation metric: how much better the model aligns to the *correct* beat versus a random
one. A larger positive gap means the model genuinely responds to the beat signal.
Examples shown here are the 5 validation windows with the highest GT−shuf gap.
""")

    gr.Markdown("---")
    gr.Markdown(
        "Each example is a 12-second window from a validation recording. "
        "Hear the beat pattern or the real vocalist first, then play the generated output."
    )

    with gr.Row():
        ql_example   = gr.Dropdown(EXAMPLE_NAMES, value=EXAMPLE_NAMES[0], label="Recording")
        ql_condition = gr.Dropdown(CONDITION_NAMES, value=CONDITION_NAMES[0], label="Beat condition")
        ql_subdiv    = gr.Dropdown(list(SUBDIVISIONS.keys()), value="Beats only",
                                   label="Click subdivisions")
    with gr.Row():
        ql_beat_btn = gr.Button("Hear beat pattern")
        ql_gt_btn   = gr.Button("Real vocalist")
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
    ql_beat_btn.click(hear_beat,     inputs=[ql_example, ql_subdiv], outputs=[ql_beat_audio])
    ql_gt_btn.click(hear_gt_vocal,   inputs=[ql_example],            outputs=[ql_gt_audio])
    ql_play_btn.click(quick_listen,
                      inputs=[ql_example, ql_condition, ql_clicks, ql_subdiv],
                      outputs=[ql_audio, ql_plot])

if __name__ == "__main__":
    demo.launch()
