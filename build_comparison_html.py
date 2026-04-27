#!/usr/bin/env python3
"""Build a self-contained HTML with base64-embedded audio."""
import base64, json, os

BASE = "/home/vm2426/beat-conditioned-GaMaDHaNi/outputs_cfg_prime"

WINDOWS = [
    ("21007_1200", 15, 0.80),
    ("20002_0",    25, 0.48),
    ("21017_1200", 15, 0.80),
    ("20002_1200", 25, 0.48),
    ("21017_0",    10, 1.20),
    ("21007_0",    10, 1.20),
]

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def audio_tag(path):
    data = b64(path)
    return f'<audio controls style="width:200px"><source src="data:audio/wav;base64,{data}" type="audio/wav"></audio>'

def get_metrics(window):
    m = json.load(open(f"{BASE}/{window}/metrics.json"))
    return m

rows_html = ""
for window, n_beats, interval in WINDOWS:
    m = get_metrics(window)
    ng  = m["noprime_gt_gs3.0"]["f1"]
    pg  = m["prime400_gt_gs3.0"]["f1"]
    nz_f1 = m["noprime_zero"]["f1"]
    gap_np = ng - m["noprime_shuf_gs3.0"]["f1"]  # for reference only

    d = f"{BASE}/{window}"

    pz_f1 = m.get("prime400_zero", {}).get("f1", None)

    noprime_gt    = audio_tag(f"{d}/audio_noprime_gt_gs3.0_click.wav")
    prime400_gt   = audio_tag(f"{d}/audio_prime400_gt_gs3.0_click.wav")
    noprime_zero  = audio_tag(f"{d}/audio_noprime_zero_click.wav")
    pz_path = f"{d}/audio_prime400_zero_click.wav"
    prime400_zero = audio_tag(pz_path) if os.path.exists(pz_path) else \
        '<span style="color:#aaa;font-size:0.8em">not generated</span>'

    pz_badge = f'<span class="badge zero">prime_zero {pz_f1:.2f}</span>' if pz_f1 is not None else ""

    rows_html += f"""
    <tr>
      <td class="row-label">
        <div class="window-id">{window}</div>
        <div class="meta">{n_beats} beats &nbsp;·&nbsp; {interval:.2f} s/beat</div>
        <div>
          <span class="badge gt">noprime GT {ng:.2f}</span>
          <span class="badge gap">gap {gap_np:+.2f}</span>
        </div>
        <div>
          <span class="badge prime">prime GT {pg:.2f}</span>
          <span class="badge zero">zero {nz_f1:.2f}</span>
          {pz_badge}
        </div>
      </td>
      <td class="col-beat">{noprime_gt}</td>
      <td class="col-beat">{prime400_gt}</td>
      <td class="col-uncon">{noprime_zero}</td>
      <td class="col-uncon">{prime400_zero}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Beat-Conditioned GaMaDHaNi — Prime Comparison</title>
<style>
  body {{ font-family: sans-serif; font-size: 14px; background: #fafafa; color: #222; margin: 0; padding: 24px; }}
  h1 {{ font-size: 1.3em; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin: 0 0 20px; font-size: 0.85em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; vertical-align: middle; }}
  .group-header {{ font-size: 1em; font-weight: bold; padding: 10px; }}
  .col-beat  {{ background: #eff6ff; }}
  .col-uncon {{ background: #fefce8; }}
  .sub-header {{ font-size: 0.85em; font-weight: 600; color: #444; background: #f0f0f0; }}
  .row-label {{ text-align: left; min-width: 160px; }}
  .window-id {{ font-weight: bold; font-family: monospace; font-size: 1em; }}
  .meta {{ font-size: 0.8em; color: #666; margin: 2px 0 4px; }}
  .badge {{ display: inline-block; font-size: 0.75em; border-radius: 3px; padding: 1px 5px; margin: 1px; }}
  .gt    {{ background: #bbf7d0; color: #065f46; }}
  .gap   {{ background: #dbeafe; color: #1e40af; }}
  .prime {{ background: #ede9fe; color: #5b21b6; }}
  .zero  {{ background: #f3f4f6; color: #374151; }}
  audio  {{ display: block; margin: 4px auto 0; width: 200px; }}
  .note  {{ font-size: 0.8em; color: #888; margin-top: 14px; }}
</style>
</head>
<body>

<h1>Beat-Conditioned GaMaDHaNi — Prime Comparison</h1>
<p class="subtitle">
  Model: 12s CFG gs=3.0 &nbsp;·&nbsp; Click tracks at GT beat positions &nbsp;·&nbsp;
  6 windows selected for audible tempo (≥ 0.48 s/beat) &nbsp;·&nbsp;
  Beat prime excluded (synthesised prime is ~95% silence)
</p>

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:left">Window</th>
      <th colspan="2" class="group-header col-beat">Beat Conditioned (GT beats)</th>
      <th colspan="2" class="group-header col-uncon">Unconditioned (zero beats)</th>
    </tr>
    <tr>
      <th class="sub-header col-beat">No prime</th>
      <th class="sub-header col-beat">Pitch prime</th>
      <th class="sub-header col-uncon">No prime</th>
      <th class="sub-header col-uncon">Pitch prime</th>
    </tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>

<p class="note">
  <b>Badges:</b> green = noprime GT F1 · blue = GT−shuffled gap · purple = pitch prime GT F1 · grey = zero-conditioned F1<br>
  <b>Columns:</b> "No prime" = generation from noise; "Pitch prime" = first 400 frames seeded with GT pitch (prime400).<br>
  <b>Unconditioned</b> = beat conditioning set to zero (null signal), not shuffled mismatched beats.
</p>

</body>
</html>"""

out = "/home/vm2426/beat-conditioned-GaMaDHaNi/prime_comparison.html"
with open(out, "w") as f:
    f.write(html)
print(f"Written: {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
