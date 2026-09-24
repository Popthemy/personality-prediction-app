# PANDORA Big Five experiments on Google Colab

Open [pandora_colab_experiments.ipynb](pandora_colab_experiments.ipynb) in Google Colab and select a GPU runtime. Run its cells in order. The notebook uses the Django-free experiment runner in this repository to train all eight combinations of Lasso/LSTM, baseline/Q-learning selection, and GAN/no GAN.

## Project code

The notebook defaults to the `pandora-3` Git branch. That branch must contain the shared-threshold changes before Colab clones it. As an alternative, copy the updated repository to `MyDrive/personality-prediction-app`; the notebook uses that Drive copy when present.

## Dataset

Put all three files in `MyDrive/pandora_personality/data/`:

- `Training dataset.xlsx`
- `Validation dataset.xlsx`
- `Test dataset.xlsx`

If none of the Excel files are present, the notebook tries to download **all three** corresponding PANDORA Big Five parquet splits from Hugging Face. It stops if any split is missing. The runner samples up to `sample_n_users` eligible training users for the smoke test, but uses **every eligible validation user** to calculate the BFI means and shared thresholds. The test split remains separate.

## Threshold and result rules

The runner takes the mean normalized ground-truth BFI score of each trait across the full eligible validation split, then averages the five means. It creates five thresholds 0.1 apart, centered on that overall mean where possible and shifted inside 0.1–0.9 otherwise. Every threshold is applied to all five traits in all eight experiments. At each threshold, the same value binarizes the ground-truth and predicted scores.

Each binary metric—Accuracy, Precision, Recall (Sensitivity), Specificity, and F1-Score—has five tables and one consolidated graph. Every table and graph labels its 0–1 score scale. MAE, RMSE, and R² are reported as continuous metrics without thresholds.

## Saved results

Each run gets a timestamped folder in `MyDrive/pandora_personality/artifacts/`. The main exports are:

| File | Contents |
|---|---|
| `shared_thresholds.json` | Trait means, overall mean, rule, and five values |
| `binary_metric_tables.xlsx` | All 25 tables on one **All metrics** worksheet |
| `plots/accuracy_by_threshold.png` and four companion images | One graph for each binary metric, with eight experiment lines averaged over traits |
| `threshold_sweeps_long.csv` | Per-experiment, per-trait results at every threshold |
| `run_summary.json` | Full recorded run results and configuration |
| `comparison.csv` | Eight-condition summary |

The notebook displays the tables and graphs. Its final cell downloads the Excel workbook and all five graph PNGs. Copies remain on Drive.

## Reruns

The prepared dataset and BERT embeddings are cached on Drive. `sample_n_users=20` is a training smoke test; increase it in the configuration cell for the reporting run. The validation means still use the full eligible validation split in either case.
