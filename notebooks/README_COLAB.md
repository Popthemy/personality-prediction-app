# PANDORA-big5 personality pipeline on Google Colab (Django-free)

Run the project's core ML pipeline on a **Colab GPU**, against the
[PANDORA-big5](https://huggingface.co/datasets/jingjietan/pandora-big5) dataset — **no Django, no
web app**. The notebook clones this repo's `pandora` branch, imports only the standalone ML
service classes, and drives them with a lightweight **`ExperimentRunner`** that stands in for the
Django orchestrator (which is coupled to the ORM and is *not* used here).

**Notebook:** [`pandora_colab_experiments.ipynb`](pandora_colab_experiments.ipynb)

---

## What it computes

A full **2×2×2 factorial** — both models run in **every** cell, so the model comparison is fair:

| factor | levels |
|---|---|
| comment selection | baseline-select · **Q-learning**-select |
| augmentation | no-GAN · **GAN** (real adversarial GAN, train-fold only) |
| model | **Lasso** (regressor) · **LSTM** (3-class sequence) |

→ **8 conditions.** From them the notebook answers three questions with matched-pair analysis:

1. **Which model is better — Lasso or LSTM?** → `model_comparison`
2. **Does Q-learning comment selection help?** → `factor_effects['qlearning_effect']`
3. **Does GAN augmentation help?** → `factor_effects['gan_effect']`

Every condition is scored on one **shared metric** — tertile Low/Med/High **accuracy + macro-F1** —
so Lasso and LSTM are directly comparable. All metrics come from the project's own
`metrics_engine`; `hybrid_cell_evaluations` additionally reports the canonical
`metrics_engine.evaluate` per matched cell (Lasso regression MAE/R²/Pearson vs LSTM tertile
accuracy/F1, plus a threshold-sweep baseline).

**Pipeline per proxy-user:** comments → (Q-learning *or* baseline) selection → BERT embedding →
(optional GAN augmentation of the *train fold only*) → Lasso **or** LSTM → OCEAN prediction.
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
9. **Claims + evidence** — findings sentences, `model_comparison`, `factor_effects`,
   `hybrid_cell_evaluations`, and figures (colorblind-safe: Lasso = blue, LSTM = orange).
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
├── artifacts/                # metrics, tables, saved models  (= cfg.output_dir)
│   ├── comparison.csv               # all 8 conditions × headline metrics
│   ├── model_comparison.csv         # Lasso vs LSTM at each matched cell
│   ├── qlearning_effect.csv         # Q-learning matched-pair deltas
│   ├── gan_effect.csv               # GAN matched-pair deltas
│   ├── findings.json                # ready-to-cite headline claims
│   ├── hybrid_evaluation.json       # metrics_engine.evaluate per matched cell
│   ├── run_summary.json             # config + sample + per-condition results
│   ├── q_table.json                 # trained Q-learning policy
│   └── <condition>/                 # e.g. lstm_qlearn_gan/
│       ├── metrics.json             # full per-trait metrics
│       ├── lasso_state.json         # (Lasso conditions)
│       └── lstm_state.pt            # (LSTM conditions)
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
- One shared train/val split and one Q-learning policy are reused across all 8 conditions, so
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
- Inspect one condition: `bundle['results']['lstm_qlearn_gan']['per_trait']['Openness']`.

**Session hygiene**
- Prefer `git pull` (cell 2 does this automatically) over re-cloning.
- Expect free-tier disconnects — because data, embeddings, and artifacts live on Drive, recovery is
  just a top-to-bottom re-run.

---

## Note on the two GANs

The Colab runner binds the **real adversarial GAN** at
`backend/ml_pipeline/services/augmentation/gan.py`. The Django production `PipelineOrchestrator`
currently uses the simpler MVP augmenter at `backend/ml_pipeline/services/gan_augmenter.py`
(Gaussian-noise, numpy-only). When the pipeline results are folded back into the Django app, that
orchestrator should be switched to the same real GAN so production matches what these experiments
measured.
