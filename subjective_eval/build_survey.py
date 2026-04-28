"""Build an HTML MOS listening-test survey from prepared samples.

Reads the directory structure produced by ``prepare_listening_test.py``
(``mos_samples/``) and the companion ``manifest.json``.

Produces:
    survey_order.csv    — maps trial_id -> system / class / sample path
    listening_test.html — self-contained HTML page with linked audio
"""

import argparse
import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = Path(__file__).resolve().parent / "mos_samples"
EBIRD_TO_ID_PATH = ROOT / "data" / "birdclef_segments" / "ebird_to_id.json"
OUTPUT_DIR = Path(__file__).resolve().parent

SPECIES_COMMON_NAMES = {
    "baffal1": "Barred Forest-Falcon",
    "bobfly1": "Boat-billed Flycatcher",
    "coffal1": "Collared Forest-Falcon",
    "compot1": "Common Potoo",
    "greant1": "Great Antshrike",
    "pirfly1": "Piratic Flycatcher",
    "roahaw": "Roadside Hawk",
    "socfly1": "Social Flycatcher",
    "trokin": "Tropical Kingbird",
    "trsowl": "Tropical Screech-Owl",
    "whtdov": "White-tipped Dove",
}

MOS_LABELS = {
    1: "Bad",
    2: "Poor",
    3: "Fair",
    4: "Good",
    5: "Excellent",
}


def _rel(path):
    return str(path.relative_to(OUTPUT_DIR))


def load_manifest(samples_dir):
    with open(samples_dir / "manifest.json") as f:
        return json.load(f)


def build_trials(samples_dir, manifest, seed=42):
    with open(EBIRD_TO_ID_PATH) as f:
        classes = json.load(f)

    rng = random.Random(seed)
    class_blocks = []

    for code in sorted(classes):
        species_dir = samples_dir / code
        if not species_dir.exists():
            print(f"  WARNING: missing species directory for {code}")
            continue

        ref_path = species_dir / "reference.wav"
        if not ref_path.exists():
            print(f"  WARNING: missing reference for {code}")
            continue

        species_map = manifest.get(code, {})

        sample_slots = [f"sample_{i}" for i in range(1, 5)]
        rng.shuffle(sample_slots)

        trials = []
        for slot in sample_slots:
            sample_path = species_dir / f"{slot}.wav"
            if not sample_path.exists():
                print(f"  WARNING: missing {sample_path}")
                continue
            trials.append(
                {
                    "ebird_code": code,
                    "common_name": SPECIES_COMMON_NAMES.get(code, code),
                    "slot": slot,
                    "system": species_map.get(slot, "unknown"),
                    "ref_path": _rel(ref_path),
                    "sample_path": _rel(sample_path),
                }
            )

        class_blocks.append(
            {
                "ebird_code": code,
                "common_name": SPECIES_COMMON_NAMES.get(code, code),
                "ref_path": _rel(ref_path),
                "trials": trials,
            }
        )

    return class_blocks


def write_csv(class_blocks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tid = 0
    rows = []
    for block in class_blocks:
        for t in block["trials"]:
            tid += 1
            rows.append(
                {
                    "trial_id": tid,
                    "ebird_code": t["ebird_code"],
                    "common_name": t["common_name"],
                    "system": t["system"],
                    "sample_path": t["sample_path"],
                    "ref_path": t["ref_path"],
                }
            )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "trial_id",
                "ebird_code",
                "common_name",
                "system",
                "sample_path",
                "ref_path",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Survey order CSV: {path}  ({tid} trials)")
    return tid


def _mos_radios(name):
    items = []
    for v in range(1, 6):
        items.append(
            f'<label class="mos-option" title="{MOS_LABELS[v]}">'
            f'<input type="radio" name="{name}" value="{v}" required>'
            f"<span>{v}</span></label>"
        )
    return "\n                  ".join(items)


def write_html(class_blocks, total_trials, path):
    trials_html = ""
    tid = 0
    for block in class_blocks:
        trials_html += f"""
    <div class="class-block">
      <h2>{block["common_name"]} <span class="ebird-code">({block["ebird_code"]})</span></h2>
      <div class="reference-section">
        <div class="audio-label">Reference recording (~35 s)</div>
        <audio controls preload="none"><source src="{block["ref_path"]}" type="audio/wav"></audio>
      </div>
      <div class="samples">"""

        for t in block["trials"]:
            tid += 1
            trials_html += f"""
        <div class="trial" id="trial-{tid}">
          <h3>Sample {tid}</h3>
          <div class="sample-audio">
            <audio controls preload="none"><source src="{t["sample_path"]}" type="audio/wav"></audio>
          </div>
          <div class="rating-row">
            <span class="rating-label">Rate the naturalness of this sample:</span>
            <div class="mos-scale">
              <span class="mos-anchor">1 Bad</span>
              {_mos_radios(f"q{tid}")}
              <span class="mos-anchor">5 Excellent</span>
            </div>
          </div>
        </div>"""

        trials_html += """
      </div>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bird Vocalization MOS Evaluation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 900px; margin: 0 auto; padding: 2rem; background: #f8f9fa; color: #212529;
         line-height: 1.5; }}
  h1 {{ margin-bottom: 0.3rem; font-size: 1.6rem; }}
  h2 {{ margin: 2rem 0 1rem; padding-bottom: 0.4rem; border-bottom: 2px solid #1976d2; }}
  .ebird-code {{ font-weight: normal; color: #6c757d; font-size: 0.9em; }}

  .instructions {{ background: #e3f2fd; border-left: 4px solid #1976d2; padding: 1rem 1.2rem;
                   margin: 1rem 0 2rem; border-radius: 4px; }}
  .instructions p {{ margin-bottom: 0.6rem; }}
  .instructions strong {{ color: #0d47a1; }}

  .audio-label {{ font-size: 0.85rem; font-weight: 600; color: #495057; margin-bottom: 0.3rem; }}
  audio {{ width: 100%; }}

  .reference-section {{ background: #fff3e0; border: 1px solid #ffe0b2; border-radius: 8px;
                        padding: 1rem; margin-bottom: 1.5rem; }}
  .reference-section audio {{ margin-top: 0.3rem; }}

  .class-block {{ margin-bottom: 3rem; }}
  .samples {{ display: flex; flex-direction: column; gap: 1rem; }}

  .trial {{ background: white; border: 1px solid #dee2e6; border-radius: 8px; padding: 1.2rem; }}
  .trial h3 {{ margin-bottom: 0.8rem; font-size: 0.95rem; color: #495057; }}
  .sample-audio {{ margin-bottom: 0.8rem; }}

  .rating-row {{ margin-top: 0.8rem; }}
  .rating-label {{ display: block; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem; }}
  .mos-scale {{ display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }}
  .mos-anchor {{ font-size: 0.75rem; color: #6c757d; min-width: 70px; }}
  .mos-option {{ cursor: pointer; }}
  .mos-option span {{ display: inline-block; padding: 0.4rem 0.8rem; border: 1px solid #ced4da;
                      border-radius: 4px; font-size: 0.9rem; text-align: center; min-width: 2.5rem;
                      transition: background 0.15s, color 0.15s; }}
  .mos-option:hover span {{ background: #e9ecef; }}
  .mos-option input {{ display: none; }}
  .mos-option input:checked + span {{ background: #1976d2; color: white; border-color: #1565c0; }}

  .submit-row {{ text-align: center; margin: 2rem 0; }}
  button[type="submit"] {{ background: #1976d2; color: white; border: none; padding: 0.9rem 2.5rem;
                           font-size: 1rem; border-radius: 6px; cursor: pointer; }}
  button[type="submit"]:hover {{ background: #1565c0; }}

  #progress {{ position: sticky; top: 0; background: white; padding: 0.5rem 1rem;
               border-bottom: 1px solid #dee2e6; z-index: 10; text-align: center;
               font-weight: 600; font-size: 0.95rem; }}
</style>
</head>
<body>

<h1>Bird Vocalization — MOS Evaluation</h1>

<div class="instructions">
  <p><strong>Thank you for participating!</strong> This listening test evaluates
  bird vocalizations.</p>

  <p>For each of <strong>11 bird species</strong> you will hear:</p>
  <ul style="margin: 0.4rem 0 0.6rem 1.5rem;">
    <li>A <strong>reference recording</strong> (~35 s) of the species for context</li>
    <li><strong>4 samples</strong> to rate individually</li>
  </ul>

  <p>Rate each sample on a <strong>1 to 5</strong> scale for how natural it sounds:</p>
  <table style="margin: 0.5rem 0 0.5rem 1rem; border-collapse: collapse; font-size: 0.9rem;">
    <tr><td style="padding: 0.15rem 0.8rem; font-weight: 600;">1</td><td>Bad</td></tr>
    <tr><td style="padding: 0.15rem 0.8rem; font-weight: 600;">2</td><td>Poor</td></tr>
    <tr><td style="padding: 0.15rem 0.8rem; font-weight: 600;">3</td><td>Fair</td></tr>
    <tr><td style="padding: 0.15rem 0.8rem; font-weight: 600;">4</td><td>Good</td></tr>
    <tr><td style="padding: 0.15rem 0.8rem; font-weight: 600;">5</td><td>Excellent</td></tr>
  </table>

  <p>Use headphones if possible. You may replay any clip as many times as you like.</p>
</div>

<div id="progress">Progress: 0 / {total_trials} rated</div>

<form id="survey-form">
{trials_html}

<div class="submit-row">
  <button type="submit">Submit Ratings</button>
</div>
</form>

<script>
const form = document.getElementById('survey-form');
const progress = document.getElementById('progress');
const total = {total_trials};

form.addEventListener('change', () => {{
  let complete = 0;
  for (let i = 1; i <= total; i++) {{
    if (form.querySelector(`input[name="q${{i}}"]:checked`)) complete++;
  }}
  progress.textContent = `Progress: ${{complete}} / ${{total}} rated`;
}});

form.addEventListener('submit', (e) => {{
  e.preventDefault();
  const results = [];
  for (let i = 1; i <= total; i++) {{
    const sel = form.querySelector(`input[name="q${{i}}"]:checked`);
    if (!sel) {{
      alert(`Please rate all samples (missing sample ${{i}})`);
      document.getElementById(`trial-${{i}}`).scrollIntoView({{ behavior: 'smooth' }});
      return;
    }}
    results.push({{ trial_id: i, mos: parseInt(sel.value) }});
  }}
  const blob = new Blob([JSON.stringify(results, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'mos_responses.json';
  a.click();
  alert('Responses saved! Thank you for participating.');
}});
</script>
</body>
</html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(html)
    print(f"HTML listening test: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build MOS listening-test HTML from prepared samples"
    )
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = load_manifest(args.samples_dir)
    print(f"Manifest loaded: {len(manifest)} species entries")

    class_blocks = build_trials(args.samples_dir, manifest, seed=args.seed)
    total_trials = sum(len(b["trials"]) for b in class_blocks)
    print(f"Class blocks: {len(class_blocks)}, trials: {total_trials}")

    write_csv(class_blocks, OUTPUT_DIR / "survey_order.csv")
    write_html(class_blocks, total_trials, OUTPUT_DIR / "listening_test.html")


if __name__ == "__main__":
    main()
