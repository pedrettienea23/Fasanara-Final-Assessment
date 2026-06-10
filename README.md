# Fasanara Capital - AI Credit Risk Assessment

A prototype credit risk assessment pipeline that estimates default probabilities for SME borrowers using structured financial data, qualitative business narratives, and machine learning. Submitted as part of the Fasanara AI Credit Analyst technical assessment.

---

## Approach Overview

The solution is structured as a three-stage pipeline:

1. Data preprocessing and feature engineering (`pipeline/01_data_processing.py`)
2. Model training, calibration, and prediction scoring (`pipeline/02_modeling.py`)
3. Analyst-style explanation generation (`pipeline/03_generate_explanations.py`)

An interactive Streamlit dashboard (`app.py`) provides a visual interface over the outputs.

---

## Pipeline

### Stage 1 - Data Processing

Three source tables are joined on `company_id`: structured financial indicators, binary default labels, and free-text business descriptions.

**Text feature extraction.** Rather than using an LLM or embedding model for the narrative descriptions, exact string matching is applied to a curated vocabulary of 10 risk phrases and 7 strength phrases drawn directly from the dataset. Each phrase becomes a binary flag (e.g. `flag_risk_declining_revenue`). This approach is interpretable, deterministic, and requires no external API. It is weaker than a semantic model but eliminates any dependency on inference latency or cost during scoring.

**Financial interaction features.** Six ratio-based features are engineered from the raw financial inputs. The rationale is that individual variables such as `debt_ratio` and `ebitda_margin` carry less signal in isolation than their interactions. For example, `distress_score = (debt_ratio / interest_coverage) - ebitda_margin - cash_ratio` compresses the four most predictive leverage and liquidity signals into a single composite. Other engineered features include debt-to-earnings, net leverage, and a liquidity coverage buffer. These interactions are standard in credit analysis practice and improve model performance without introducing data leakage.

**Categorical encoding.** `sector` and `country` are one-hot encoded with `drop_first=True` to avoid perfect multicollinearity. The scoring set is aligned to the exact column schema of the training set after encoding, with any unseen categories filled with zero.

---

### Stage 2 - Modeling

**Scaling.** A `RobustScaler` is applied exclusively to the 15 continuous numeric features, using the median and interquartile range rather than the mean and standard deviation. This makes scaling resistant to outliers, which is important in financial data where single extreme observations in debt or coverage ratios are common. Binary flags and one-hot encoded columns are passed through unchanged.

**Baseline model.** A Logistic Regression with `class_weight="balanced"` serves as the interpretable baseline. The balanced weighting adjusts for the approximately 1:5 class imbalance (defaulters represent around 17% of the training set) by up-weighting the minority class during optimisation. Logistic Regression also outputs probabilities that are already reasonably calibrated by construction, making it a meaningful comparison point.

**Primary model - XGBoost.** An XGBoost classifier is selected as the primary model for its ability to capture non-linear interactions between financial ratios and qualitative flags without requiring explicit feature crosses. The `scale_pos_weight` parameter is set to the observed class imbalance ratio (4.75) so the boosting process assigns proportionally higher loss to missed defaults. Hyperparameters are selected via 5-fold stratified grid search optimising Average Precision (PR-AUC), which is more appropriate than ROC-AUC under class imbalance because it conditions performance on the positive class.

Best parameters: `max_depth=3`, `learning_rate=0.05`, `n_estimators=100`, `subsample=0.8`, `colsample_bytree=0.8`. Shallow trees with a low learning rate reduce overfitting; column subsampling introduces regularisation similar to Random Forest.

**Probability calibration.** The raw XGBoost model produces well-ranked predictions but miscalibrated probabilities. In credit risk, probability values matter directly: they feed into expected loss calculations (`EL = PD x LGD x EAD`). A model that outputs 35% when the true probability is 8% would systematically overprice loans or misallocate capital reserves. Platt scaling (sigmoid calibration) is applied using `CalibratedClassifierCV(method='sigmoid', cv=5)`. This fits a two-parameter sigmoid function on held-out fold predictions to map raw scores to calibrated probabilities.

**Calibration quality metrics.** Standard metrics (ROC-AUC, Brier Score) do not fully separate calibration from discrimination. Two additional metrics are tracked:

- **Expected Calibration Error (ECE)**: probabilities are binned into 10 equal-width buckets; for each bucket the absolute gap between mean predicted probability and observed positive rate is calculated and weighted by bin size. A perfectly calibrated model scores 0. The calibrated XGBoost achieves ECE of 2.62%, down from 16.13% on the raw model.
- **Probability MAD (pMAD)**: the mean absolute deviation of predicted probabilities from their own mean. A model that assigns every observation the base rate (around 17%) would be trivially well-calibrated but useless for discrimination. pMAD catches this failure; the calibrated model maintains a pMAD of 11.15%, confirming the probabilities are diverse.
- **ECE/pMAD ratio**: the ratio of miscalibration to diversity. Lower is better. The calibrated model achieves 0.2349, versus 0.7412 for raw XGBoost and 0.9296 for Logistic Regression.

**Decision threshold tuning.** At the default 0.5 threshold, the calibrated XGBoost misses 87% of actual defaults (FNR 87%), prioritising precision over recall. In credit lending, a missed default (false negative) represents a capital loss; a false approval is far more costly than a false rejection. The threshold is therefore lowered to 0.0982, chosen by finding the operating point that achieves a target recall of 86.8% (FNR 13.2%) on out-of-fold validation. At this threshold, FPR rises to approximately 50%, meaning roughly half of safe borrowers are incorrectly referred for review. This is an acceptable trade-off given the asymmetric cost structure.

**Risk tiers.** Probabilities are assigned to three tiers: Low (below half the operating threshold), Medium (between half the threshold and the threshold), and High (at or above the threshold). High corresponds to a recommended decline.

---

### Stage 3 - Explanations

SHAP TreeExplainer is used to produce local feature attributions for each scored company. SHAP values decompose the model output into additive contributions from each feature, guaranteeing consistency with the model's actual decision boundary (unlike simpler approximation methods such as LIME).

The explanation generator identifies the top three risk drivers (positive SHAP values) and top three mitigating factors (negative SHAP values) for each company and assembles them into a readable one-paragraph summary. An optional LLM path exists (OpenAI GPT-4o or Anthropic Claude, triggered by setting an API key in `.env`) that uses the SHAP context as a structured prompt to produce more fluent prose. In the absence of an API key, the local template-based fallback is used, which is fully deterministic and produces consistent output.

---

## Model Performance Summary

| Model | Threshold | Brier Score | ECE | pMAD | ECE/pMAD | Recall | FPR | FNR |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.1862 | 21.32% | 22.93% | 0.930 | 64.94% | 25.91% | 35.06% |
| XGBoost (tuned, raw) | 0.50 | 0.1570 | 16.13% | 21.76% | 0.741 | 63.22% | 20.46% | 36.78% |
| XGBoost (Platt calibrated) | 0.0982 | 0.1192 | 2.62% | 11.15% | 0.235 | 86.78% | 49.76% | 13.22% |

---

## Repository Structure

```
.
├── data/
│   ├── train_companies.csv
│   ├── train_narratives.csv
│   ├── train_outcomes.csv
│   ├── scoring_companies.csv
│   └── processed/           # Generated by pipeline/01
├── pipeline/
│   ├── 01_data_processing.py
│   ├── 02_modeling.py
│   └── 03_generate_explanations.py
├── outputs/
│   └── predictions.csv      # Final scored output
├── app.py                   # Streamlit dashboard
└── README.md
```

---

## Running the Pipeline

Dependencies are managed with `uv`. All commands should be run from the repository root.

```bash
# 1. Process data and engineer features
uv run python pipeline/01_data_processing.py

# 2. Train models, calibrate, and generate scored predictions
uv run python pipeline/02_modeling.py

# 3. Generate analyst-style explanations and write outputs/predictions.csv
uv run python pipeline/03_generate_explanations.py

# 4. (Optional) Launch the interactive Streamlit dashboard
uv run streamlit run app.py
```

No API key is required to run the full pipeline. The LLM explanation path is an optional enhancement; set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in a `.env` file and set `GENERATE_EXPLANATIONS = True` in `pipeline/03_generate_explanations.py` to activate it.

---

## Design Decisions and Limitations

**Why XGBoost over alternatives.** Random Forest and Gradient Boosting methods were tested in early exploratory work. LightGBM and HistGradientBoosting produce comparable results but XGBoost provides the most stable TreeExplainer integration with SHAP, which was a priority given the explanation requirement.

**Why Platt scaling over isotonic regression.** Isotonic regression is a non-parametric calibration method that can overfit on small datasets. With 1,000 training observations and 5-fold cross-validation, Platt scaling is the safer choice. It also produces a smooth, monotonic probability mapping, which is preferable for downstream use in expected loss calculations.

**Why ECE and pMAD in addition to Brier Score.** Brier Score conflates calibration and discrimination into a single number. ECE isolates calibration; pMAD isolates diversity. Tracking both ensures the model is not rewarded for collapsing all predictions toward the base rate. The ECE/pMAD ratio provides a single ranking criterion for comparing models on the calibration dimension specifically.

**Text features.** The current text feature approach is a deliberate simplification. A proper NLP pipeline using sentence embeddings (e.g. `sentence-transformers`) or a fine-tuned language model would capture richer signal from the narratives. The tradeoff is that the current approach is fully auditable, requires no inference infrastructure, and is transparent enough to present to a risk committee.

**Threshold selection.** The 13.2% FNR operating point was chosen as a defensible balance between capital protection and operational throughput. In production, this threshold would be calibrated against actual loss-given-default estimates and the cost of manual review for false positives.