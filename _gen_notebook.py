"""Build the current PANDORA Google Colab notebook from the project runner."""

import json
from pathlib import Path


OUT = Path("notebooks/pandora_colab_experiments.ipynb")
CELLS = []


def md(source):
    CELLS.append(("markdown", source))


def code(source):
    CELLS.append(("code", source))


md("""# PANDORA Big Five experiments on Google Colab

Run the eight Lasso/LSTM × baseline/Q-learning × GAN/no-GAN experiments with the same five validation-derived thresholds. Each threshold is used for all five traits in every experiment. The run saves 25 binary metric tables in one Excel worksheet and five PNG graphs, one per metric.

**Threshold rule:** compute each trait's mean normalized BFI score across the entire eligible validation file, average those five means, and make five thresholds 0.1 apart. Center them on the overall mean when possible; shift the set within 0.1–0.9 otherwise. At each threshold, the same boundary binarizes ground truth and prediction. MAE, RMSE, and R² remain continuous metrics.

Run the cells in order on a GPU runtime. Put the three PANDORA Excel files in `MyDrive/pandora_personality/data/` before starting the data cell. If those files are absent, the notebook looks for all three split parquet files on Hugging Face. A single unsplit file is deliberately insufficient for this workflow.
""")

md("## 1. Mount Drive and set paths")
code("""from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/pandora_personality')
DATA_DIR = DRIVE_ROOT / 'data'
CACHE_DIR = DRIVE_ROOT / 'cache'
ARTIFACT_DIR = DRIVE_ROOT / 'artifacts'
for directory in (DATA_DIR, CACHE_DIR, ARTIFACT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
print('Data folder:', DATA_DIR)
""")

md("""## 2. Load the project code

The default branch below is the branch with the shared-threshold implementation. Push the local changes to that branch before cloning in Colab, or copy the updated project into `MyDrive/personality-prediction-app` to use the Drive copy.
""")
code("""import subprocess, sys

REPO_URL = 'https://github.com/Popthemy/personality-prediction-app.git'
REPO_BRANCH = 'pandora-3'
DRIVE_REPO = DRIVE_ROOT.parent / 'personality-prediction-app'
REPO_DIR = DRIVE_REPO if DRIVE_REPO.is_dir() else Path('/content/personality-prediction-app')

if REPO_DIR != DRIVE_REPO:
    if not REPO_DIR.exists():
        subprocess.run(['git', 'clone', '--branch', REPO_BRANCH, '--single-branch',
                        REPO_URL, str(REPO_DIR)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', REPO_BRANCH], check=True)
        subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', REPO_BRANCH], check=True)
        subprocess.run(['git', '-C', str(REPO_DIR), 'pull', '--ff-only', 'origin', REPO_BRANCH], check=True)

assert (REPO_DIR / 'backend/ml_pipeline/experiments/result_exports.py').is_file(), (
    'This checkout lacks the shared-threshold export. Update the GitHub branch or Drive copy.'
)
sys.path.insert(0, str(REPO_DIR))
print('Using project:', REPO_DIR)
""")

md("## 3. Install dependencies and check the GPU")
code("""subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r',
                str(REPO_DIR / 'requirements-colab.txt')], check=True)
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
""")

md("""## 4. Prepare the train, validation, and test files

Preferred files in the Drive data folder: `Training dataset.xlsx`, `Validation dataset.xlsx`, and `Test dataset.xlsx`. If none are present, the notebook downloads the corresponding PANDORA Big Five parquet splits. The full eligible validation file is used for the BFI means; `sample_n_users` controls only training sampling.
""")
code("""import os, shutil

excel_names = {
    'train': 'Training dataset.xlsx',
    'validation': 'Validation dataset.xlsx',
    'test': 'Test dataset.xlsx',
}
excel_paths = {split: DATA_DIR / name for split, name in excel_names.items()}
present = {split: path.exists() for split, path in excel_paths.items()}
if any(present.values()) and not all(present.values()):
    raise FileNotFoundError('Supply all three split Excel files in ' + str(DATA_DIR))

if all(present.values()):
    destination = REPO_DIR / 'PANDORA'
    destination.mkdir(parents=True, exist_ok=True)
    for split, source in excel_paths.items():
        target = destination / excel_names[split]
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        print(split, target)
else:
    from huggingface_hub import list_repo_files, hf_hub_download
    repo_id = 'jingjietan/pandora-big5'
    available = list_repo_files(repo_id, repo_type='dataset')
    destination = REPO_DIR / 'PANDORA/pandora-big5/data'
    destination.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation', 'test'):
        names = [name for name in available if Path(name).name.startswith(split + '-')
                 and name.endswith('.parquet')]
        if not names:
            raise FileNotFoundError(f'No {split} parquet split was found. Upload all three Excel files to {DATA_DIR}.')
        for name in names:
            source = Path(hf_hub_download(repo_id, name, repo_type='dataset',
                                          local_dir=str(DATA_DIR / 'pandora-big5')))
            target = destination / Path(name).name
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        print(split, len(names), 'parquet shard(s)')

os.chdir(REPO_DIR)  # The split loader resolves PANDORA/ relative to the project root.
""")

md("## 5. Configure and inspect the complete validation split")
code("""from backend.ml_pipeline.experiments.pandora_runner import (
    ExperimentConfig, ExperimentRunner, load_file_defined_splits)
from backend.ml_pipeline.services.metrics_engine import shared_thresholds_from_validation

cfg = ExperimentConfig(
    sample_n_users=20,  # training smoke test; raise for the reporting run
    min_comments_per_user=5,
    top_k=10,
    qlearning_train_epochs=3,
    lstm_epochs=35,
    seed=42,
    embedding_cache_dir=str(CACHE_DIR),
    output_dir=str(ARTIFACT_DIR),
)
splits = load_file_defined_splits(work_dir=DRIVE_ROOT, cfg=cfg)
threshold_plan = shared_thresholds_from_validation(
    splits.sample.labels_unit[splits.val_idx])
print('Eligible users — train:', len(splits.train_idx),
      'validation:', len(splits.val_idx), 'test:', len(splits.test_idx))
print('Validation trait means:', threshold_plan['trait_means'])
print('Overall mean:', round(threshold_plan['overall_mean'], 6))
print('Shared thresholds:', threshold_plan['thresholds'])
""")

md("""## 6. Run all eight experiments

The same validation-derived threshold set is used for every trait and condition. The runner saves results to a timestamped folder on Drive, including the Excel workbook and five graph images.
""")
code("""runner = ExperimentRunner(splits, cfg)
bundle = runner.run()
run_dir = Path(bundle['artifact_dir'])
print('Saved run:', run_dir)
print('Thresholds:', bundle['shared_thresholds']['thresholds'])
display(bundle['comparison'])
""")

md("## 7. View the 25 binary metric tables")
code("""from IPython.display import display, Markdown
from backend.ml_pipeline.experiments.result_exports import METRICS, EXPERIMENT_ORDER
from backend.ml_pipeline.experiments.pandora_runner import OCEAN_TRAITS, TRAIT_KEYS
import pandas as pd

sweep = bundle['threshold_sweeps']
condition_order = [key for _, key in EXPERIMENT_ORDER]
trait_aliases = dict(zip(OCEAN_TRAITS, TRAIT_KEYS))
for title, key in METRICS:
    display(Markdown('### ' + title + ' — metric score scale: 0–1 (0.8 = 80%)'))
    for threshold in bundle['shared_thresholds']['thresholds']:
        display(Markdown(f'**{title} at threshold {threshold:.3f}** — score scale: 0–1'))
        block = sweep[sweep['threshold'].sub(threshold).abs() < 1e-8].copy()
        block['trait'] = block['trait'].replace(trait_aliases)
        table = block.pivot(index='trait', columns='condition', values=key)
        table = table.reindex(index=['O', 'C', 'A', 'N', 'E'], columns=condition_order)
        table.columns = [label for label, _ in EXPERIMENT_ORDER]
        display(table.round(4))
""")

md("## 8. View the five consolidated graphs and continuous metrics")
code("""from IPython.display import Image

for title, key in METRICS:
    display(Markdown(f'### {title} — metric score scale: 0–1 (0.8 = 80%)'))
    display(Image(filename=str(run_dir / 'plots' / f'{key}_by_threshold.png')))

for title, key, direction in (('MAE', 'val_mae', 'min'),
                              ('RMSE', 'val_rmse', 'min'),
                              ('R²', 'val_r2', 'max')):
    values = {condition: result['overall'][key]
              for condition, result in bundle['results'].items()
              if result['overall'].get(key) is not None}
    winner = (min if direction == 'min' else max)(values, key=values.get)
    print(f'{title} is a continuous regression metric and has no threshold. '
          f'{winner} has the best result: {values[winner]:.4f}.')
""")

md("""## 9. Verify and download the exports

The workbook has one `All metrics` worksheet containing the 25 tables. Each graph is a separate PNG. All six files are already on Drive; run the final cell to download copies to your computer.
""")
code("""workbook = run_dir / 'binary_metric_tables.xlsx'
images = [run_dir / 'plots' / f'{key}_by_threshold.png' for _, key in METRICS]
assert workbook.is_file() and all(path.is_file() for path in images)
assert bundle['shared_thresholds']['thresholds'] == threshold_plan['thresholds']
for result in bundle['results'].values():
    for block in result['validation']['per_trait'].values():
        actual = [row['threshold'] for row in block['threshold_sweep']]
        assert actual == threshold_plan['thresholds'], actual
print('Verified: one common threshold list across 8 experiments and 5 traits.')
print('Excel:', workbook)
for image in images:
    print('Image:', image)
""")

code("""from google.colab import files
files.download(str(workbook))
for image in images:
    files.download(str(image))
""")


def main():
    cells = []
    for index, (kind, source) in enumerate(CELLS):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
            "id": f"pandora-{index:02d}",
        }
        if kind == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    notebook = {
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
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT}: {len(cells)} cells")


if __name__ == "__main__":
    main()
