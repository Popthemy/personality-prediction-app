# PANDORA Architecture And Training Pipeline Notes

## Current Goal

The experiment predicts Big Five personality traits from user comments and compares eight conditions:

- Lasso vs LSTM
- baseline comment selection vs Q-learning comment selection
- no GAN vs GAN augmentation

The active classification result is binary Low/High per trait. Low/Medium/High bands are still measured, but they are used for imbalance reporting and targeted augmentation, not as the final classifier output.

## Data Split Flow

The pipeline now prefers the existing file-defined datasets:

- `Training dataset.xlsx` is used for model fitting only.
- `Validation dataset.xlsx` is used to choose model decision thresholds.
- `Test dataset.xlsx` is used for final reported performance and prediction evaluation profiles.

This is important because threshold choice is a modeling decision. If the test set chooses its own threshold, the test score becomes optimistic. The test set should only answer: "How well did the already-chosen model and threshold work?"

The Excel files include the real `author` field. The pipeline now groups comments by author, so each profile represents one author and no longer relies on the old trait-tuple proxy that was needed when the parquet export did not expose usernames.

## Threshold Flow

There are two different thresholds in the pipeline:

- Trait-label threshold: converts the true continuous OCEAN score into Low or High.
- Model decision threshold: converts the model prediction into Low or High.

The trait-label threshold is now derived from the training split only, separately for O, C, E, A, and N, using the train median for each trait. That gives each trait a data-backed Low/High boundary without looking at validation or test labels.

The model decision threshold is selected on validation predictions, separately for each trait and condition. Candidate thresholds are derived from the validation prediction-score distribution for that trait, then the chosen threshold maximizes the harmonic mean of F1 and specificity. The selected validation threshold is then locked and applied once to the test split.

## Imbalance Flow

The pipeline now reports Low/Medium/High distribution for train, validation, and test. Training sample weights are also derived from the train split so underrepresented trait bands carry more weight during model fitting.

This does not change the task into three-class classification. It means the model remains Low/High, while the training process becomes more aware of whether the data is concentrated around certain trait regions.

## Targeted GAN Flow

GAN augmentation now focuses on minority trait-band rows from the training split:

- The train labels are divided into Low/Medium/High bands for each trait.
- Rows belonging to underrepresented bands are selected as the GAN fitting pool.
- If there are enough minority-band rows, GAN trains on that minority-focused pool.
- If there are not enough rows, GAN safely falls back or skips augmentation.

This makes GAN's role specific: it supports scarce trait regions instead of generating synthetic examples uniformly across the whole dataset.

## Model Roles

Lasso is the simpler baseline model. It uses mean-pooled text embeddings and trains one regularized regression model per trait. It is useful because it is stable, interpretable, and gives a sober reference point for whether the deeper model is earning its complexity.

LSTM is the sequence model. It reads comment embeddings as an ordered sequence per user and predicts all five OCEAN traits jointly. It can capture patterns across multiple comments that a simple pooled model may flatten.

Q-learning is the comment selector. It learns which comments are more useful to keep before feature extraction. It is trained only on the training split so the selector does not learn from validation or test users.

Q-learning should be defended in two separate ways. First, compare its model-quality effect against the full-history baseline using the matched `qlearning_effect.csv` rows. Second, compare its practical selection cost using `qlearning_efficiency.json`. The baseline now embeds every available comment for an author. Q-learning sees that same full candidate pool, then uses the learned select/skip policy to choose a smaller subset before BERT. This lets the project report both predictive performance and estimated compute savings from fewer BERT embedding calls.

GAN is the augmenter. It generates synthetic embedding-and-trait examples for training only. Its job is not to prove final performance by itself; its valid use is to improve coverage of underrepresented trait regions and then let validation/test metrics decide whether that helped.

## Reporting Contract

After a run, the important audit artifacts are:

- `imbalance_report.json`: trait-band distribution by split.
- `trait_label_thresholds.json`: train-derived Low/High boundaries per trait.
- `targeted_gan_report.json`: whether GAN targeted minority bands and how many synthetic rows/sequences it generated.
- `qlearning_efficiency.json`: comments selected, BERT calls avoided versus full author histories, and Q-learning's matched accuracy/F1 deltas.
- `threshold_sweeps_long.csv`: validation/test threshold evidence.
- `prediction_evidence.csv`: condition-level prediction results.

Together, these files explain what data was used, how imbalance was handled, how thresholds were selected, and which model condition actually performed best.
