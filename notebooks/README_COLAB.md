# PANDORA-big5 personality pipeline on Google Colab (Django-free)

Run the project's core ML pipeline on a **Colab GPU**, against the
[PANDORA-big5](https://huggingface.co/datasets/jingjietan/pandora-big5) dataset — **no Django, no
web app**. The notebook clones this repo's `pandora` branch, imports only the standalone ML
service classes, and drives them with a lightweight **`ExperimentRunner`** that stands in for the
Django orchestrator (which is coupled to the ORM and is *not* used here).

**Notebook:** [`pandora_colab_experiments.ipynb`](pandora_colab_experiments.ipynb)

---

## What it computes

A full **2×2×2 factorial** across the supervisor-requested pipeline combinations:

| factor | levels |
|---|---|
| comment selection | baseline-select · **Q-learning**-select |
| augmentation | no-GAN · **GAN** (real adversarial GAN, train-fold only) |
| final model | **Lasso** · **LSTM binary classifier** |

→ **8 conditions.** From them the notebook answers three questions with matched-pair analysis:

1. **Does Q-learning comment selection help?** → `factor_effects['qlearning_effect']`
2. **Does GAN augmentation help?** → `factor_effects['gan_effect']`
3. **Which final model performs better, Lasso or LSTM?** → `factor_effects['model_comparison']`

Every condition is scored with the same binary Low/High metrics:
**accuracy, precision, recall, F1, specificity, ROC-AUC, and PR-AUC**. Lasso emits five normalized
continuous OCEAN scores; LSTM emits five probabilities, one P(High) for each OCEAN trait. The
validation split chooses the best decision threshold per trait from `0.30, 0.40, 0.50, 0.60, 0.70`;
the test split then applies those frozen validation-selected thresholds once.

**Pipeline per proxy-user:** comments → (Q-learning *or* baseline) selection → BERT embedding →
(optional GAN augmentation of the *train fold only*) → Lasso or LSTM final model → Low/High OCEAN
prediction.
One PANDORA proxy-user (a unique `(O,C,E,A,N)` tuple) = one "volunteer".

---

## How to open it

- **From GitHub:** in Colab → *File → Open notebook → GitHub*, enter this repo, pick branch
  `pandora`, open `notebooks/pandora_colab_experiments.ipynb`. **Or** prefix the file's GitHub URL
  with `https://colab.research.google.com/github/`.
- **By upload:** download the `.ipynb` and *File → Upload notebook* in Colab.

Then **Runtime → Change runtime type → GPU**, and run top to bottom.

---

## Run flow (the notebook's sections)

1. **Mount Drive** — lays out `MyDrive/pandora_personality/{data,cache,artifacts}`.
2. **Get the code** — clones/pulls the `pandora` branch, puts the repo root on `sys.path`
   (Drive-upload fallback documented in-cell if the remote isn't reachable).
3. **Install** `requirements-colab.txt` — the minimal Django-free deps (Colab already ships
   torch/numpy/pandas/sklearn).
4. **GPU check** — `nvidia-smi` + `torch.cuda.is_available()`.
5. **Download PANDORA** parquet(s) from the HuggingFace Hub → Drive (idempotent/cached).
6. **EDA** — confirms the O/C/E/A/N label scale the runner auto-detects and maps to `[0,1]`.
7. **Load → clean → sample + configure** — groups comments per proxy-user, runs the project's
   cleaner, caches the cleaned result to Drive; builds `ExperimentConfig`.
8. **Run the sweep** — `runner = ExperimentRunner(prepared, cfg); bundle = runner.run()`.
9. **Claims + evidence** — findings sentences, `comparison`, `factor_effects`, prediction-level
   evidence, metric audit, per-trait threshold sweeps, model comparison, and figures.
10. **Artifacts on Drive** + reload demo.
11. **Debugging & tips.**

Start with the built-in **20-user smoke test** (`sample_n_users=20`) to validate the whole flow
end-to-end in minutes, then raise `sample_n_users` (and `lstm_epochs`) for the reporting run — a
one-line change in the config cell.

---

## Drive layout

```
MyDrive/pandora_personality/
├── data/                     # PANDORA parquet(s) from HuggingFace
├── cache/                    # BERT embedding cache (.npy by text hash)
├── artifacts/                # run archive parent (= cfg.output_dir)
│   └── run_YYYYMMDD_HHMMSS/         # one timestamped folder per full run
│       ├── comparison.csv           # all 8 conditions × headline metrics
│       ├── presentation_metrics_long.csv # long-form metric table for charts/BI tools
│       ├── threshold_sweeps_long.csv # all 5 thresholds × traits × conditions
│       ├── prediction_evidence.csv  # per-user truth, score, threshold, predicted label, outcome
│       ├── classification_audit.json # recomputes metrics from evidence; must PASS
│       ├── artifact_manifest.json   # presentation order, threshold policy, graph policy
│       ├── qlearning_effect.csv     # Q-learning matched-pair deltas
│       ├── gan_effect.csv           # GAN matched-pair deltas
│       ├── model_comparison.csv     # LSTM-vs-Lasso matched-pair deltas
│       ├── findings.json            # ready-to-cite headline claims
│       ├── run_summary.json         # config + sample + per-condition results
│       ├── q_table.json             # trained Q-learning policy
│       ├── plots/                   # presentation-ready PNG metric + threshold graphs
│       └── <condition>/             # e.g. lstm_qlearn_gan/
│           ├── metrics.json         # full per-trait metrics
│           ├── lasso_state.json     # trained Lasso state for Lasso conditions
│           └── lstm_state.pt        # trained LSTM state for LSTM conditions
└── pandora_prepared.json     # cleaned + grouped users (cached)
```

Everything expensive to recompute lives on Drive, so a runtime disconnect never forces a re-encode
or re-clean — just re-run top to bottom and the caches make it quick.

---

## Tips

**Speed / caching**
- **BERT encoding dominates.** Embeddings are cached to `cache/` on Drive keyed by text hash — the
  first run is slow, every re-run is fast.
- **Cleaned data is cached too** (`pandora_prepared.json`), so you skip re-cleaning the parquet.
- Iterate with the **20-user smoke test**; scale `sample_n_users` / `lstm_epochs` up for the report.

**GPU**
- Verify the GPU cell shows `CUDA available: True` before the sweep. If not:
  **Runtime → Change runtime type → GPU**.
- **OOM?** Lower `bert_max_length` (256 → 128) and/or `lstm_batch_size` in the config cell.

**Reproducibility**
- `seed` in `ExperimentConfig` drives the sample, the split, Q-learning exploration, and LSTM init
  (the runner calls `set_seed` internally). Same seed + same cache ⇒ same numbers.
- One shared train/val/test split and one Q-learning policy are reused across all 8 conditions, so
  differences reflect the **factor under study**, not split luck.

**Debugging**
- `logging.getLogger('ml_pipeline').setLevel(logging.DEBUG)` for verbose progress.
- `%debug` in a fresh cell for a post-mortem into the last exception.
- Run the runner's pieces one at a time on the objects already in memory, e.g.:
  ```python
  from backend.ml_pipeline.experiments import pandora_runner as R
  sample = R.sample_users(prepared, cfg)
  enc = R.get_encoder()
  feats = R.build_features(sample, 'qlearning', cfg, enc, agent=R.train_qlearning_agent(sample, cfg))
  ```
- Inspect one condition: `bundle['results']['lstm_qlearn_gan']['test']['per_trait']['O']`.

**Session hygiene**
- Prefer `git pull` (cell 2 does this automatically) over re-cloning.
- Expect free-tier disconnects — because data, embeddings, and artifacts live on Drive, recovery is
  just a top-to-bottom re-run.

---

## Note on Lasso and LSTM

Lasso and LSTM are both final-model conditions in this experiment. Lasso is trained on pooled BERT
embeddings and evaluated as a thresholded continuous scorer; LSTM is trained on selected-comment
embedding sequences and evaluated as a binary classifier. Both use the same participant split and
the same validation-selected threshold policy.
