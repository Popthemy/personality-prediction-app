# Project Updates & Modifications - Developer: Ajadi

This document lists the design changes, refactors, and feature additions implemented by **Ajadi** for the Big Five Personality Prediction platform. 

---

## 1. On-Demand X Profile Analyzer Feature

### [NEW] X Profile Analyze Page
- **Template**: Created [`analyze.html`](file:///c:/Users/USER/Demola/personality-prediction-app/backend/templates/tools/analyze.html)
- **Styling**: Vanilla modern CSS featuring a premium dark-amber glassmorphic UI, responsive grids, soft gradients, and custom active state transitions matching the researcher portal aesthetics.
- **Loader Interface**: Designed a procedural step-by-step spinner overlay that lists pipeline execution stages in real-time.

### [NEW] BFI-44 Questionnaire Wizard
- **Template**: Created [`bfi_q.html` Snippet](file:///c:/Users/USER/Demola/personality-prediction-app/backend/templates/tools/snippets/bfi_q.html)
- **Flow**: Built a 4-page paginated wizard using lightweight, smooth JS state transitions to handle self-rating questionnaires without cluttering the screen.
- **Likert Selector**: Styled custom circular 1-5 rating selectors with active state glowing borders for a clean user experience.

### [NEW] Backend Control Flow Check
- **Location**: Modified [`views.py`](file:///c:/Users/USER/Demola/personality-prediction-app/backend/tools/views.py) in the `AnalyzeProfileView` class.
- **Dynamic Check**: Integrates a two-step routing conditional block:
  - If a submitted X handle already exists with ground truth BFI data, proceeds directly to fetch, pre-process, embed, and run Lasso prediction metrics.
  - If no ground truth data is present, the view redirects the flow, rendering the BFI-44 Survey wizard.
  - Upon survey submission, calculates OCEAN traits using `BFIScorer`, stores a `BFI_SURVEY` ground-truth object mapped to the volunteer, and triggers the active ML sequence.

### [NEW] Navigation Links
- Added the "Analyze Profile" route tab inside [`base.html`](file:///c:/Users/USER/Demola/personality-prediction-app/backend/templates/base.html) for seamless accessibility from all pages.

---

## 2. ML Pipeline Integration & Alignment

### Text Preprocessing Integration
- Refactored `_step_1_input_data`, `_step_2_qlearning_selection`, and `_step_3_bert_embedding` in the pipeline orchestration code to ensure raw posts fetch and clean mentions or links first, using `TextPreprocessor` before passing them to the Q-Learning selection and HuggingFace BERT encoder layers.

### Idempotent Embedding Caching
- Configured check cycles within the prediction stack. BERT embeddings check for preexisting `BERT_EMBEDDING` database records by post reference to reduce server CPU load and API network traffic.

---

## 3. Template bug Fixes & Enhancements

### Resolved Structural Variable Bugs
- Debugged template inconsistencies in [`volunteer_detail.html`](file:///c:/Users/USER/Demola/personality-prediction-app/backend/templates/dashboard/volunteer_detail.html) and [`index.html`](file:///c:/Users/USER/Demola/personality-prediction-app/backend/templates/dashboard/index.html) to align matches with database names:
  * Renamed `mae_score` $\to$ `overall_mae`
  * Renamed `openness_predicted` $\to$ `predicted_openness`
  * Renamed `conscientiousness_predicted` $\to$ `predicted_conscientiousness`
  * Renamed `extraversion_predicted` $\to$ `predicted_extraversion`
  * Renamed `agreeableness_predicted` $\to$ `predicted_agreeableness`
  * Renamed `neuroticism_predicted` $\to$ `predicted_neuroticism`
  * Checked OneToOne relationships to ensure ground truth matches are checked as `volunteer.bfi_survey` rather than `volunteer.bfi_surveys.first`.

### Domain Recommendation Engine
- Made the **Education**, **Health**, and **Employment** advisory sections in [`volunteer_detail.html`](file:///c:/Users/USER/Demola/personality-prediction-app/backend/templates/dashboard/volunteer_detail.html) dynamic. The insight text now generates based on the volunteer's predicted Big Five score thresholds (e.g. recommending research roles for high Openness, stress management for high Neuroticism).

---

## 4. Dependencies
- Installed and appended to [`requirements.txt`](file:///c:/Users/USER/Demola/personality-prediction-app/requirements.txt):
  - `x-tweet-fetcher`: For public Nitter pipeline fetching logic.
  - `django-mathfilters`: For percentage calculations in HTML templates.
