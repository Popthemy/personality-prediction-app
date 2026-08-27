"""Generate notebooks/pandora_colab_experiments.ipynb (deleted after use)."""
import json
from pathlib import Path

OUT = Path("notebooks/pandora_colab_experiments.ipynb")

# Each entry is (cell_type, source_string). Code cell sources are PLAIN strings
# (never f-strings) so tokens like {req} and %.2f survive verbatim.
CELLS = []
def md(s): CELLS.append(("markdown", s))
def code(s): CELLS.append(("code", s))


md("""# PANDORA-big5 · Personality Pipeline on Colab (Django-free)

Runs the full **2×2×2 factorial** of the ML pipeline directly on the
[PANDORA-big5](https://huggingface.co/datasets/jingjietan/pandora-big5) dataset —
**no Django, no web app** — driving the pipeline's ML service classes from
`backend/ml_pipeline/experiments/pandora_runner.py`.

**Factors** (both models run in every cell, so the model comparison is fair):

| factor | levels |
|---|---|
| comment selection | baseline-select · **Q-learning**-select |
| augmentation | no-GAN · **GAN** |
| model | **Lasso** (regressor) · **LSTM** (3-class sequence) |

→ **8 conditions.** The notebook then answers three questions with matched-pair analysis:

1. **Which model is better — Lasso or LSTM?**  → `model_comparison`
2. **Does Q-learning comment selection help?**  → `factor_effects['qlearning_effect']`
3. **Does GAN augmentation help?**  → `factor_effects['gan_effect']`

Every condition is scored on one **shared metric** — tertile Low/Med/High **accuracy + macro-F1** —
so Lasso and LSTM are directly comparable; Lasso additionally reports regression MAE/R²/Pearson.

**Pipeline per user:** comments → (Q-learning *or* baseline) selection → BERT embedding →
(optional GAN augmentation of the *train fold only*) → Lasso **or** LSTM → OCEAN prediction.

> **Run top to bottom.** Set **Runtime → Change runtime type → GPU** first. The dataset, the
> BERT embedding cache, and all artifacts persist to Google Drive, so re-runs are near-instant
> and survive disconnects. The final cell has debugging tips.""")

md("## 1 · Mount Google Drive")
code("""# Mount Drive and lay out the project folders on it. Everything expensive to
# recompute (dataset, BERT embeddings, artifacts) lives on Drive, so it survives
# runtime disconnects and makes re-runs fast.
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_ROOT = Path('/content/drive/MyDrive/pandora_personality')
except ModuleNotFoundError:
    # Not on Colab — fall back to a local folder so the notebook still runs.
    DRIVE_ROOT = Path('./pandora_personality').resolve()
    print('google.colab not found — using local folder:', DRIVE_ROOT)

DATA_DIR      = DRIVE_ROOT / 'data'          # PANDORA parquet(s) from HuggingFace
CACHE_DIR     = DRIVE_ROOT / 'cache'         # BERT embedding cache (.npy by text hash)
ARTIFACT_DIR  = DRIVE_ROOT / 'artifacts'     # metrics, comparison tables, saved models
PREPARED_JSON = DRIVE_ROOT / 'pandora_prepared.json'   # cleaned+grouped users (cached)
for d in (DATA_DIR, CACHE_DIR, ARTIFACT_DIR):
    d.mkdir(parents=True, exist_ok=True)
print('Drive project root:', DRIVE_ROOT)""")

md("""## 2 · Get the code onto the runtime

Default: clone (or `git pull`) the repo and put its root on `sys.path`. If your remote isn't
reachable from Colab, use the **Drive-upload fallback**: upload/copy the repo folder to
`MyDrive/personality-prediction-app`, then set `REPO_DIR = DRIVE_ROOT.parent / 'personality-prediction-app'`
and skip the clone.""")
code("""import sys, subprocess

REPO_URL    = 'https://github.com/Popthemy/personality-prediction-app.git'
REPO_BRANCH = 'pandora'
REPO_DIR    = Path('/content/personality-prediction-app')

if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', '--branch', REPO_BRANCH, '--single-branch',
                    REPO_URL, str(REPO_DIR)], check=True)
else:
    subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only'], check=False)

# The repo root (folder that CONTAINS `backend/`) must be on sys.path so that
# `import backend.ml_pipeline...` resolves. `backend` is a namespace package.
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
print('repo on sys.path:', REPO_DIR, '| has backend/:', (REPO_DIR / 'backend').is_dir())""")

md("## 3 · Install the minimal (Django-free) dependencies")
code("""# Colab already ships torch/numpy/pandas/scikit-learn with a CUDA-matched torch,
# so this typically only adds transformers, huggingface_hub and pyarrow. The file
# uses loose lower bounds on purpose — never hard-pin torch on Colab or you may
# replace its GPU build with a CPU-only wheel.
req = REPO_DIR / 'requirements-colab.txt'
!pip install -q -r "{req}"
print('installed from', req)""")

md("## 4 · Confirm the GPU")
code("""# If this reports CPU only, set Runtime → Change runtime type → GPU and re-run.
!nvidia-smi -L || echo "no nvidia-smi (CPU runtime)"
import torch
print('torch', torch.__version__,
      '| CUDA available:', torch.cuda.is_available(),
      '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')""")

md("""## 5 · Download PANDORA-big5 from the HuggingFace Hub → Drive

Cached: `hf_hub_download` is idempotent, so this only downloads on the first run. If the dataset
is ever gated, run `from huggingface_hub import login; login()` with your token first.""")
code("""from huggingface_hub import hf_hub_download, list_repo_files

HF_REPO = 'jingjietan/pandora-big5'
all_files = list_repo_files(HF_REPO, repo_type='dataset')
parquet_files = ([f for f in all_files if f.endswith('.parquet') and f.startswith('data/')]
                 or [f for f in all_files if f.endswith('.parquet')])
print('parquet files in repo:', parquet_files)

local_parquets = []
for rel in parquet_files:
    p = hf_hub_download(HF_REPO, rel, repo_type='dataset', local_dir=str(DATA_DIR))
    local_parquets.append(Path(p))
    print('ready:', Path(p).name)

# A small fixed sample only needs one shard; use the first.
PANDORA_FILE = local_parquets[0]
print('\\nusing PANDORA file:', PANDORA_FILE)""")

md("""## 6 · EDA — confirm the label scale

The runner auto-detects the O/C/E/A/N range and maps it to `[0,1]`. Seeing the real min/max here
makes that mapping **auditable** (e.g. `[0,100]` → `v/100`, `[1,5]` → `(v-1)/4`).""")
code("""import pandas as pd
df = pd.read_parquet(PANDORA_FILE, columns=['O','C','E','A','N','ptype','text'])
print('rows:', len(df), '| columns:', list(df.columns))
display(df[['O','C','E','A','N']].describe().T[['min','max','mean','std']])
print('\\nexample comment:', df['text'].iloc[0][:200])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.0))
for tr in ['O', 'C', 'E', 'A', 'N']:
    ax.hist(df[tr].dropna(), bins=40, histtype='step', label=tr, linewidth=1.5)
ax.set_title('O/C/E/A/N distribution (raw scale)')
ax.set_xlabel('raw trait value'); ax.set_ylabel('count')
ax.legend(ncol=5, fontsize=8, frameon=False)
plt.tight_layout(); plt.show()""")

md("""## 7 · Load → clean → sample, and configure the run

`load_pandora_comments` groups comments by the `(O,C,E,A,N)` proxy-user key and runs the
project's `DataCleaner`. We cache the cleaned result to Drive and reload it on later runs to
skip re-cleaning the whole parquet.""")
code("""import json
from types import SimpleNamespace
from backend.ml_pipeline.services.data.pandora import (
    load_pandora_comments, PreparedUserComments, UserTraits)
from backend.ml_pipeline.experiments import ExperimentConfig, ExperimentRunner, OCEAN_TRAITS

def load_prepared_cache(path):
    \"\"\"Rebuild the prepared users from the cached JSON. Downstream code only needs
    `.cleaned_text` per comment, so lightweight namespaces are enough.\"\"\"
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    out = []
    for u in data:
        traits = UserTraits(**u['traits']) if u['traits'] else None
        comments = [SimpleNamespace(**c) for c in u['comments']]  # each has .cleaned_text
        out.append(PreparedUserComments(user_id=u['user_id'], traits=traits, comments=comments))
    return out

if 'prepared' not in globals():
    if PREPARED_JSON.exists():
        prepared = load_prepared_cache(PREPARED_JSON)
        print('loaded cached prepared users from', PREPARED_JSON)
    else:
        prepared = load_pandora_comments(str(PANDORA_FILE), output_path=str(PREPARED_JSON))
print(f'{len(prepared)} proxy-users | '
      f'{sum(len(u.comments) for u in prepared)} cleaned comments total')

cfg = ExperimentConfig(
    sample_n_users=20,        # 20-user smoke test → fast end-to-end check; raise (e.g. 60) for the reporting run
    min_comments_per_user=5,
    top_k=10,                 # comments kept per user (selection budget)
    qlearning_train_epochs=3,
    val_ratio=0.2,
    lstm_epochs=35,           # lower (e.g. 10) for quick iteration
    seed=42,
    embedding_cache_dir=str(CACHE_DIR),   # persist BERT embeddings to Drive
    output_dir=str(ARTIFACT_DIR),         # persist metrics + models to Drive
)
cfg""")

md("""## 8 · Run the 8-condition factorial sweep

First run encodes the BERT embeddings (cached to Drive by text hash). Later runs reuse the cache
and are dramatically faster. **`ExperimentRunner`** is the lightweight, Django-free stand-in for
the pipeline's orchestrator — it drives the *same* service classes (BERT / Q-learning / GAN /
Lasso / LSTM / metrics_engine) over in-memory PANDORA data, with **no ORM**. Its `.run()` samples
once, trains one Q-learning agent, builds the baseline & Q-learning features once, uses **one
shared train/val split**, then runs all 8 conditions and saves artifacts to Drive.""")
code("""import logging
logging.getLogger('ml_pipeline').setLevel(logging.INFO)   # use DEBUG for more detail

# ExperimentRunner replaces the Django orchestrator; it returns plain dicts + DataFrames.
runner = ExperimentRunner(prepared, cfg)
bundle = runner.run()          # sample → 8 conditions → analyze → save to Drive
runner.comparison              # the 8-condition headline table (same as bundle['comparison'])""")

md("""## 9 · The claims, with the matched-pair evidence

`findings.notes` are ready-to-cite sentences; the tables below are the evidence behind them.""")
code("""print('=== FINDINGS ===')
for note in bundle['findings']['notes']:
    print(' -', note)

print('\\n=== Lasso vs LSTM  (matched selection × GAN cells) ===')
display(bundle['model_comparison'])
print('=== Q-learning effect  (delta vs baseline-select) ===')
display(bundle['factor_effects']['qlearning_effect'])
print('=== GAN effect  (delta vs no-GAN) ===')
display(bundle['factor_effects']['gan_effect'])""")

md("""### The canonical `metrics_engine` view (Lasso vs LSTM per cell)

`hybrid_cell_evaluations` is the project's own `metrics_engine.evaluate` applied to each matched
(selection × GAN) cell — the **same evaluation module the Django app uses**. It scores Lasso by
regression (MAE / R² / Pearson) and LSTM by 3-class tertile (accuracy / F1), alongside a
threshold-sweep baseline, so the head-to-head rests on the pipeline's authoritative metrics — not
only the runner's shared-metric table above. This is also written to `hybrid_evaluation.json` on
Drive.""")
code("""# One row per matched cell: Lasso vs LSTM as scored by metrics_engine.evaluate.
rows = []
for cell, ev in bundle['hybrid_cell_evaluations'].items():
    la = ev.get('lasso', {}).get('aggregate', {})
    ls = ev.get('lstm',  {}).get('aggregate', {})
    th = ev.get('threshold', {}).get('aggregate', {})
    rows.append({
        'cell': cell,
        'lasso_MAE': la.get('mae'), 'lasso_R2': la.get('r2'), 'lasso_r': la.get('correlation'),
        'lstm_acc': ls.get('accuracy'), 'lstm_F1': ls.get('f1'),
        'thr_acc': th.get('accuracy'), 'thr_F1': th.get('f1'),
    })
hybrid_df = pd.DataFrame(rows)
display(hybrid_df.round(4))""")

md("""### Figures

Colorblind-safe palette (validated): **Lasso = blue, LSTM = orange** in fixed order; effect bars
diverge **blue = helps / red = hurts** around zero. Every chart carries a legend and direct value
labels, so identity is never conveyed by color alone. One y-axis per chart; recessive gridlines.""")
code("""import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- light theme with recessive chrome (validated palette) ---
SURFACE='#fcfcfb'; INK='#0b0b0b'; INK2='#52514e'; MUTED='#898781'; GRID='#e1e0d9'; AXIS='#c3c2b7'
rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'text.color': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlecolor': INK,
    'font.size': 10, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'figure.dpi': 120,
})
LASSO_C='#2a78d6'; LSTM_C='#eb6834'      # categorical slots 1,2 (fixed order)
POS_C='#2a78d6';   NEG_C='#d03b3b'       # diverging: blue=helps, red=hurts

# The four matched cells (selection, gan, short label), in a fixed reading order.
CELLS = [('baseline', False, 'baseline'), ('qlearning', False, 'Q-learn'),
         ('baseline', True,  'baseline+GAN'), ('qlearning', True, 'Q-learn+GAN')]

def overall(model, sel, gan, metric):
    for r in bundle['results'].values():
        if r['model']==model and r['selection']==sel and bool(r['gan'])==gan:
            return r['overall'][metric]
    return None

# Figure 1 — Lasso vs LSTM at each matched cell, on the two shared metrics.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
labels = [c[2] for c in CELLS]; x = np.arange(len(CELLS)); w = 0.38
for ax, metric, title in [(axes[0], 'accuracy', 'Tertile accuracy'),
                          (axes[1], 'macro_f1', 'Macro-F1')]:
    la = [overall('lasso', s, g, metric) for s, g, _ in CELLS]
    ls = [overall('lstm',  s, g, metric) for s, g, _ in CELLS]
    b1 = ax.bar(x - w/2, la, w, label='Lasso', color=LASSO_C)
    b2 = ax.bar(x + w/2, ls, w, label='LSTM',  color=LSTM_C)
    ax.bar_label(b1, fmt='%.2f', padding=2, fontsize=8, color=INK2)
    ax.bar_label(b2, fmt='%.2f', padding=2, fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1); ax.set_title(title); ax.grid(axis='x', visible=False)
axes[0].set_ylabel('score (validation)')
axes[0].legend(frameon=False, ncol=2, loc='upper left')
fig.suptitle('Lasso vs LSTM at matched (selection × GAN) conditions', fontsize=12)
plt.tight_layout(); plt.show()""")

code("""# Figure 2 — isolated factor effects as matched-pair deltas (positive = helps).
def effect_panel(ax, dframe, cat_cols, title):
    rows = list(dframe.iterrows())
    labels = [' · '.join(str(r[c]) for c in cat_cols) for _, r in rows]
    vals = [r['delta_accuracy'] for _, r in rows]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=[POS_C if v >= 0 else NEG_C for v in vals])
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
    for yi, v in zip(y, vals):
        ax.text(v + (0.002 if v >= 0 else -0.002), yi, f'{v:+.3f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10); ax.grid(axis='y', visible=False)
    m = max(0.01, max(abs(v) for v in vals)); ax.set_xlim(-m*1.5, m*1.5)
    ax.set_xlabel('Δ accuracy')

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
effect_panel(axes[0], bundle['factor_effects']['qlearning_effect'], ['model', 'gan'],
             'Q-learning effect  (vs baseline-select)')
effect_panel(axes[1], bundle['factor_effects']['gan_effect'], ['model', 'selection'],
             'GAN effect  (vs no-GAN)')
fig.suptitle('Isolated factor effects — positive Δ = the factor helped', fontsize=12)
plt.tight_layout(); plt.show()""")

code("""# Figure 3 — confusion matrix for the best condition (summed over the 5 traits).
# Sequential single-hue (Blues): light = few, dark = many.
best_id = bundle['findings']['best_condition']['condition']
best = bundle['results'][best_id]
cm = np.sum([np.array(best['per_trait'][t]['confusion_matrix']) for t in OCEAN_TRAITS], axis=0)

fig, ax = plt.subplots(figsize=(4.6, 4.1))
ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(['Low', 'Med', 'High']); ax.set_yticklabels(['Low', 'Med', 'High'])
ax.set_xlabel('predicted'); ax.set_ylabel('true'); ax.grid(False)
thr = cm.max() * 0.6
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(cm[i, j]), ha='center', va='center', fontsize=11,
                color='white' if cm[i, j] > thr else INK)
ax.set_title('Confusion matrix (summed over traits)\\nbest condition: ' + best_id, fontsize=10)
plt.tight_layout(); plt.show()""")

md("""## 10 · Artifacts on Drive + reload demo

`runner.run()` wrote everything under `cfg.output_dir`. The comparison + model-comparison tables,
the factor-effect deltas, the per-condition metrics, `findings.json`, `hybrid_evaluation.json`
(the `metrics_engine` per-cell evaluation), the Q-learning policy, and each trained model (Lasso
state as JSON, LSTM `state_dict` as `.pt`) all persist to Drive.""")
code("""import json
print('artifacts under', ARTIFACT_DIR, ':')
for p in sorted(ARTIFACT_DIR.rglob('*')):
    if p.is_file():
        print('  ', p.relative_to(ARTIFACT_DIR))

print('\\ncomparison.csv reloaded from Drive:')
display(pd.read_csv(ARTIFACT_DIR / 'comparison.csv'))
print('findings.json → better_model:',
      json.loads((ARTIFACT_DIR / 'findings.json').read_text())['better_model'])""")

md("""## 11 · Debugging & tips

**Speed / caching**
- **BERT encoding dominates.** Embeddings are cached to `CACHE_DIR` on Drive keyed by text hash —
  the first run is slow, every re-run is fast. Keeping the cache on Drive means a runtime
  disconnect never forces a re-encode.
- **Cleaned data is cached too** (`pandora_prepared.json`), so you skip re-cleaning the parquet.
- Iterate fast with a **small sample**: lower `sample_n_users` and `lstm_epochs` (e.g. 10), confirm
  the whole flow, then raise them for the reporting run — one-line changes in the config cell.

**GPU**
- Verify the GPU cell shows `CUDA available: True` before the sweep. If not:
  **Runtime → Change runtime type → GPU**.
- **OOM?** Lower `bert_max_length` (256 → 128) and/or `lstm_batch_size`.

**Reproducibility**
- `seed` in `ExperimentConfig` drives the sample, the split, Q-learning exploration, and LSTM init
  (the runner calls `set_seed` internally). Same seed + same cache ⇒ same numbers.
- One shared train/val split and one Q-learning policy are reused across all 8 conditions, so
  differences reflect the **factor under study**, not split luck.

**Debugging**
- `logging.getLogger('ml_pipeline').setLevel(logging.DEBUG)` for verbose progress.
- Post-mortem into the last exception with `%debug` in a fresh cell.
- Run one piece at a time on the objects already in memory, e.g.:
  ```python
  from backend.ml_pipeline.experiments import pandora_runner as R
  sample = R.sample_users(prepared, cfg)
  enc = R.get_encoder()
  feats = R.build_features(sample, 'qlearning', cfg, enc, agent=R.train_qlearning_agent(sample, cfg))
  ```
- Inspect a single condition: `bundle['results']['lstm_qlearn_gan']['per_trait']['Openness']`.

**Session hygiene**
- Prefer `git pull` (cell 2 does this automatically) over re-cloning.
- Expect free-tier disconnects — because data, embeddings, and artifacts live on Drive, you just
  re-run top to bottom and the caches make it quick.""")


def main():
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            "id": f"cell{i:02d}",
        }
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)

    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    # Validate it parses and report shape.
    reloaded = json.loads(OUT.read_text(encoding="utf-8"))
    kinds = [c["cell_type"] for c in reloaded["cells"]]
    print("wrote", OUT, "| cells:", len(kinds),
          "| markdown:", kinds.count("markdown"), "| code:", kinds.count("code"))


if __name__ == "__main__":
    main()
