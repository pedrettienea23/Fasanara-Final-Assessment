#!/usr/bin/env python
# coding: utf-8

# # AI Credit Risk Assessment - Preprocessing & Modeling Pipeline (v2)
#
# This script implements an improved modeling approach:
# 1. Loading the preprocessed training and scoring datasets.
# 2. Applying `RobustScaler` specifically to continuous numeric features.
# 3. Applying SMOTE on training folds to address class imbalance.
# 4. Training and evaluating:
#    - Baseline: Logistic Regression (class_weight='balanced')
#    - Main: XGBoost (scale_pos_weight tuned)
#    - New: LightGBM (is_unbalance=True)
#    - Ensemble: Stacking (XGBoost + LightGBM → Logistic Regression meta-learner)
# 5. Performing post-hoc probability calibration using Platt scaling.
# 6. Optimizing decision thresholds to achieve target FNR ≤ 15%.
# 7. Generating predictions for scoring companies and calculating feature contributions using SHAP.
#

import pandas as pd
import numpy as np
import os
import pickle
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, brier_score_loss, confusion_matrix
)
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import shap

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("WARNING: imbalanced-learn not installed. SMOTE will be skipped.")

warnings.filterwarnings("ignore")

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("Imports and styling completed successfully.")
print(f"SMOTE available: {SMOTE_AVAILABLE}")


# ── 1. Load Data ──────────────────────────────────────────────────────────────

# Resolve paths relative to this script's location so it works from any CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "processed")
os.makedirs(DATA_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_processed.csv"))
score_df = pd.read_csv(os.path.join(DATA_DIR, "scoring_processed.csv"))

X = train_df.drop(columns=["defaulted", "company_id"])
y = train_df["defaulted"]

X_score = score_df.drop(columns=["company_id"])
score_company_ids = score_df["company_id"]

print(f"\nX shape: {X.shape}, y shape: {y.shape}")
print(f"X_score shape: {X_score.shape}")
print(f"Default rate: {y.mean():.3%}")


# ── 2. Preprocessor ───────────────────────────────────────────────────────────

continuous_cols = [
    "revenue_m", "ebitda_margin", "debt_ratio", "interest_coverage",
    "cash_ratio", "years_in_operation", "employee_count", "revenue_growth",
    "distress_score", "debt_cov_profitability", "debt_to_earnings",
    "debt_to_coverage", "net_leverage", "debt_margin_stress", "liquidity_coverage"
]
passthrough_cols = [col for col in X.columns if col not in continuous_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), continuous_cols),
        ("pass", "passthrough", passthrough_cols)
    ]
)

feature_names = continuous_cols + passthrough_cols
print(f"\nPreprocessor configured. Scaling {len(continuous_cols)} numeric, "
      f"passing through {len(passthrough_cols)} binary/encoded features.")


# ── 3. Evaluation Helper ──────────────────────────────────────────────────────

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    roc_auc  = roc_auc_score(y_true, y_prob)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    pr_auc   = auc(rec_arr, prec_arr)
    f1       = f1_score(y_true, y_pred, zero_division=0)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    rec      = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr      = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr      = fn / (fn + tp) if (fn + tp) > 0 else 0
    brier    = brier_score_loss(y_true, y_prob)
    return dict(roc_auc=roc_auc, pr_auc=pr_auc, f1=f1, precision=prec,
                recall=rec, fpr=fpr, fnr=fnr, brier_score=brier)


# ── 4. Logistic Regression Baseline ──────────────────────────────────────────

print("\n=== [1/4] Logistic Regression Baseline ===")

oof_probs_lr = np.zeros(len(X))
fold_metrics_lr = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    X_tr_sc = preprocessor.fit_transform(X_tr)
    X_va_sc = preprocessor.transform(X_va)

    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr_sc, y_tr = sm.fit_resample(X_tr_sc, y_tr)

    model = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(X_tr_sc, y_tr)

    probs = model.predict_proba(preprocessor.transform(X.iloc[va_idx]))[:, 1]
    oof_probs_lr[va_idx] = probs
    fold_metrics_lr.append(evaluate_predictions(y_va, probs))
    print(f"  Fold {fold+1}: ROC-AUC={fold_metrics_lr[-1]['roc_auc']:.4f}  "
          f"PR-AUC={fold_metrics_lr[-1]['pr_auc']:.4f}  "
          f"FNR={fold_metrics_lr[-1]['fnr']:.2%}  FPR={fold_metrics_lr[-1]['fpr']:.2%}")

df_lr = pd.DataFrame(fold_metrics_lr)
print("\nAverage LR Baseline:")
print(df_lr.mean().to_string())


# ── 5. XGBoost ────────────────────────────────────────────────────────────────

print("\n=== [2/4] XGBoost (scale_pos_weight) ===")

imbalance_ratio = (y == 0).sum() / (y == 1).sum()
print(f"  scale_pos_weight = {imbalance_ratio:.2f}")

XGB_PARAMS = dict(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=imbalance_ratio,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
    use_label_encoder=False,
)

oof_probs_xgb = np.zeros(len(X))
fold_metrics_xgb = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    X_tr_sc = preprocessor.fit_transform(X_tr)
    X_va_sc = preprocessor.transform(X_va)

    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr_sc, y_tr_res = sm.fit_resample(X_tr_sc, y_tr)
    else:
        y_tr_res = y_tr

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_tr_sc, y_tr_res)

    probs = model.predict_proba(X_va_sc)[:, 1]
    oof_probs_xgb[va_idx] = probs
    fold_metrics_xgb.append(evaluate_predictions(y_va, probs))
    print(f"  Fold {fold+1}: ROC-AUC={fold_metrics_xgb[-1]['roc_auc']:.4f}  "
          f"PR-AUC={fold_metrics_xgb[-1]['pr_auc']:.4f}  "
          f"FNR={fold_metrics_xgb[-1]['fnr']:.2%}  FPR={fold_metrics_xgb[-1]['fpr']:.2%}")

df_xgb = pd.DataFrame(fold_metrics_xgb)
print("\nAverage XGBoost:")
print(df_xgb.mean().to_string())


# ── 6. LightGBM ───────────────────────────────────────────────────────────────

print("\n=== [3/4] LightGBM ===")

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    is_unbalance=True,
    random_state=42,
    verbosity=-1,
)

oof_probs_lgb = np.zeros(len(X))
fold_metrics_lgb = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    X_tr_sc = preprocessor.fit_transform(X_tr)
    X_va_sc = preprocessor.transform(X_va)

    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr_sc, y_tr_res = sm.fit_resample(X_tr_sc, y_tr)
    else:
        y_tr_res = y_tr

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_tr_sc, y_tr_res)

    probs = model.predict_proba(X_va_sc)[:, 1]
    oof_probs_lgb[va_idx] = probs
    fold_metrics_lgb.append(evaluate_predictions(y_va, probs))
    print(f"  Fold {fold+1}: ROC-AUC={fold_metrics_lgb[-1]['roc_auc']:.4f}  "
          f"PR-AUC={fold_metrics_lgb[-1]['pr_auc']:.4f}  "
          f"FNR={fold_metrics_lgb[-1]['fnr']:.2%}  FPR={fold_metrics_lgb[-1]['fpr']:.2%}")

df_lgb = pd.DataFrame(fold_metrics_lgb)
print("\nAverage LightGBM:")
print(df_lgb.mean().to_string())


# ── 7. Stacking Ensemble ──────────────────────────────────────────────────────

print("\n=== [4/4] Stacking Ensemble (XGBoost + LightGBM -> LogReg meta) ===")

# Build the full pipeline preprocessor + estimator per base learner
# We do manual stacking to allow SMOTE within each fold correctly.

oof_probs_stack = np.zeros(len(X))
fold_metrics_stack = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    X_tr_sc = preprocessor.fit_transform(X_tr)
    X_va_sc = preprocessor.transform(X_va)

    # SMOTE on training fold
    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr_sc, y_tr)
    else:
        X_tr_res, y_tr_res = X_tr_sc, y_tr

    # Inner cross-validation to get base-learner OOF meta-features
    inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    meta_train = np.zeros((len(X_tr_res), 2))  # [xgb_prob, lgb_prob]

    xgb_m = xgb.XGBClassifier(**XGB_PARAMS)
    lgb_m = lgb.LGBMClassifier(**LGBM_PARAMS)

    for inner_fold, (in_tr, in_va) in enumerate(inner_skf.split(X_tr_res, y_tr_res)):
        xgb_m.fit(X_tr_res[in_tr], y_tr_res.iloc[in_tr] if hasattr(y_tr_res, 'iloc') else y_tr_res[in_tr])
        lgb_m.fit(X_tr_res[in_tr], y_tr_res.iloc[in_tr] if hasattr(y_tr_res, 'iloc') else y_tr_res[in_tr])
        meta_train[in_va, 0] = xgb_m.predict_proba(X_tr_res[in_va])[:, 1]
        meta_train[in_va, 1] = lgb_m.predict_proba(X_tr_res[in_va])[:, 1]

    # Train full base learners on all resampled training data
    xgb_full = xgb.XGBClassifier(**XGB_PARAMS)
    lgb_full = lgb.LGBMClassifier(**LGBM_PARAMS)
    xgb_full.fit(X_tr_res, y_tr_res)
    lgb_full.fit(X_tr_res, y_tr_res)

    # Build meta-features for validation
    meta_val = np.column_stack([
        xgb_full.predict_proba(X_va_sc)[:, 1],
        lgb_full.predict_proba(X_va_sc)[:, 1]
    ])

    # Train meta-learner on inner OOF predictions
    y_tr_res_arr = y_tr_res.values if hasattr(y_tr_res, 'values') else y_tr_res
    meta_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_lr.fit(meta_train, y_tr_res_arr)

    probs = meta_lr.predict_proba(meta_val)[:, 1]
    oof_probs_stack[va_idx] = probs
    fold_metrics_stack.append(evaluate_predictions(y_va, probs))
    print(f"  Fold {fold+1}: ROC-AUC={fold_metrics_stack[-1]['roc_auc']:.4f}  "
          f"PR-AUC={fold_metrics_stack[-1]['pr_auc']:.4f}  "
          f"FNR={fold_metrics_stack[-1]['fnr']:.2%}  FPR={fold_metrics_stack[-1]['fpr']:.2%}")

df_stack = pd.DataFrame(fold_metrics_stack)
print("\nAverage Stacking Ensemble:")
print(df_stack.mean().to_string())


# ── 8. Summary Comparison Table ───────────────────────────────────────────────

print("\n\n=== [OK] Cross-Validation Performance Summary (default threshold 0.5) ===")
summary = pd.DataFrame({
    "Logistic Regression": df_lr.mean(),
    "XGBoost":             df_xgb.mean(),
    "LightGBM":            df_lgb.mean(),
    "Stacking Ensemble":   df_stack.mean(),
}).T[["roc_auc", "pr_auc", "brier_score", "f1", "precision", "recall", "fpr", "fnr"]]

pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
print(summary.to_string())


# ── 9. Select Best Model at the Business Operating Point ─────────────────────
#
# Instead of using PR-AUC at the default 0.5 threshold, we select the model
# that achieves the LOWEST FPR when its threshold is tuned to keep FNR <= 15%.
# This directly aligns model selection with the business objective.

TARGET_RECALL = 0.85  # i.e., FNR <= 15%

def find_operating_point(y_true, y_probs, target_recall=0.85):
    """Find the threshold that achieves target_recall and return the resulting FPR."""
    prec_a, rec_a, thr_a = precision_recall_curve(y_true, y_probs)
    # Search from high recall side; pick threshold where recall first drops to target
    valid = rec_a[:-1] >= target_recall
    if valid.sum() == 0:
        valid = rec_a[:-1] >= 0.80  # relax if target not achievable
    f1_a = 2 * prec_a[:-1] * rec_a[:-1] / (prec_a[:-1] + rec_a[:-1] + 1e-9)
    f1_a = np.where(valid, f1_a, 0)
    idx = np.argmax(f1_a)
    thr = thr_a[idx]
    y_pred = (y_probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    f1  = f1_score(y_true, y_pred, zero_division=0)
    return dict(threshold=thr, fpr=fpr, fnr=fnr, recall=tp/(tp+fn), f1=f1, precision=tp/(tp+fp) if (tp+fp)>0 else 0)

oof_map = {
    "Logistic Regression": oof_probs_lr,
    "XGBoost":             oof_probs_xgb,
    "LightGBM":            oof_probs_lgb,
    "Stacking Ensemble":   oof_probs_stack,
}

print("\n=== Model Selection at Business Operating Point (FNR <= 15%) ===")
op_results = {}
for name, probs in oof_map.items():
    op = find_operating_point(y, probs, target_recall=TARGET_RECALL)
    op_results[name] = op
    print(f"  {name:<22}: threshold={op['threshold']:.4f}  "
          f"FNR={op['fnr']:.2%}  FPR={op['fpr']:.2%}  F1={op['f1']:.2%}")

# Choose the model with lowest FPR at the operating point
best_model_name = min(op_results, key=lambda n: op_results[n]["fpr"])
best_oof = oof_map[best_model_name]
print(f"\nBest model (lowest FPR at FNR<=15%): {best_model_name}")
print(f"  -> FPR={op_results[best_model_name]['fpr']:.2%}  "
      f"FNR={op_results[best_model_name]['fnr']:.2%}  "
      f"F1={op_results[best_model_name]['f1']:.2%}")


# ── 10. Calibrate the Best Model ─────────────────────────────────────────────

print("\n=== Calibrating Best Model with Platt Scaling ===")

oof_probs_cal = np.zeros(len(X))
fold_metrics_cal = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    X_tr_sc = preprocessor.fit_transform(X_tr)
    X_va_sc = preprocessor.transform(X_va)

    if SMOTE_AVAILABLE:
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr_sc, y_tr)
    else:
        X_tr_res, y_tr_res = X_tr_sc, y_tr

    # Train the best base model and wrap in Platt calibration
    if best_model_name == "XGBoost":
        base = xgb.XGBClassifier(**XGB_PARAMS)
    elif best_model_name == "LightGBM":
        base = lgb.LGBMClassifier(**LGBM_PARAMS)
    elif best_model_name == "Logistic Regression":
        base = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    else:  # Stacking Ensemble -> use XGBoost as base (highest PR-AUC after LR)
        base = xgb.XGBClassifier(**XGB_PARAMS)

    cal_model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    cal_model.fit(X_tr_res, y_tr_res)

    probs = cal_model.predict_proba(X_va_sc)[:, 1]
    oof_probs_cal[va_idx] = probs
    fold_metrics_cal.append(evaluate_predictions(y_va, probs))

df_cal = pd.DataFrame(fold_metrics_cal)
print("\nCalibrated model OOF metrics (default threshold):")
print(df_cal.mean().to_string())
brier_cal = brier_score_loss(y, oof_probs_cal)
print(f"\nOverall Brier Score Loss (calibrated): {brier_cal:.4f}")

# Check calibrated model at operating point
cal_op = find_operating_point(y, oof_probs_cal, target_recall=TARGET_RECALL)
print(f"Calibrated model at operating point (FNR<={int((1-TARGET_RECALL)*100)}%):")
print(f"  threshold={cal_op['threshold']:.4f}  FNR={cal_op['fnr']:.2%}  "
      f"FPR={cal_op['fpr']:.2%}  F1={cal_op['f1']:.2%}  Precision={cal_op['precision']:.2%}")

# Pick whichever is better at operating point: raw OOF or calibrated
if cal_op["fpr"] <= op_results[best_model_name]["fpr"]:
    print("\nCalibration improved FPR at operating point. Using calibrated probabilities.")
    final_oof_for_threshold = oof_probs_cal
    use_calibration = True
else:
    print("\nRaw probabilities better at operating point. Using raw OOF probabilities.")
    final_oof_for_threshold = best_oof
    use_calibration = False



# ── 11. Threshold Optimisation ───────────────────────────────────────────────

print("\n=== Threshold Optimisation -- targeting FNR <= 15% ===")

# Use whichever probability set gave better FPR at the operating point
precision_arr, recall_arr, thresholds = precision_recall_curve(y, final_oof_for_threshold)

valid_mask = recall_arr[:-1] >= TARGET_RECALL
if valid_mask.sum() == 0:
    valid_mask = recall_arr[:-1] >= 0.80

f1_scores = 2 * precision_arr[:-1] * recall_arr[:-1] / (
    precision_arr[:-1] + recall_arr[:-1] + 1e-9)
f1_valid = np.where(valid_mask, f1_scores, 0)
best_idx = np.argmax(f1_valid)
optimal_threshold = thresholds[best_idx]

y_pred_opt = (final_oof_for_threshold >= optimal_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y, y_pred_opt).ravel()
opt_rec  = tp / (tp + fn)
opt_fpr  = fp / (fp + tn)
opt_fnr  = fn / (fn + tp)
opt_f1   = f1_score(y, y_pred_opt)
opt_prec = precision_score(y, y_pred_opt, zero_division=0)

print(f"Optimal Threshold : {optimal_threshold:.4f}")
print(f"Recall (TPR)      : {opt_rec:.2%}")
print(f"FNR               : {opt_fnr:.2%}")
print(f"FPR               : {opt_fpr:.2%}")
print(f"Precision         : {opt_prec:.2%}")
print(f"F1-Score          : {opt_f1:.2%}")


# ── 12. Risk Tiering ─────────────────────────────────────────────────────────

def assign_risk_rating(prob, thr):
    if prob < thr / 2:
        return "Low"
    elif prob < thr:
        return "Medium"
    else:
        return "High"


# ── 13. Calibration Reliability Diagram ──────────────────────────────────────

prob_true_cal, prob_pred_cal = calibration_curve(y, oof_probs_cal, n_bins=10)
prob_true_raw, prob_pred_raw = calibration_curve(y, best_oof, n_bins=10)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated", color="black")
ax.plot(prob_pred_raw, prob_true_raw, marker="o", label=f"{best_model_name} (raw)", color="red")
ax.plot(prob_pred_cal, prob_true_cal, marker="s", label=f"{best_model_name} + Platt Scaling", color="green")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title(f"Reliability Diagram ({best_model_name})")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "calibration_curve.png"), dpi=150)
plt.close()
print("\nSaved calibration_curve.png")


# ── 14. Final Model Training & Scoring ───────────────────────────────────────

print("\n=== Training Final Model on All Training Data ===")

X_scaled_all = preprocessor.fit_transform(X)

if SMOTE_AVAILABLE:
    sm_final = SMOTE(random_state=42, k_neighbors=5)
    X_final_res, y_final_res = sm_final.fit_resample(X_scaled_all, y)
else:
    X_final_res, y_final_res = X_scaled_all, y

if best_model_name == "XGBoost":
    final_base = xgb.XGBClassifier(**XGB_PARAMS)
elif best_model_name == "LightGBM":
    final_base = lgb.LGBMClassifier(**LGBM_PARAMS)
elif best_model_name == "Logistic Regression":
    final_base = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
else:  # Stacking Ensemble
    final_base = xgb.XGBClassifier(**XGB_PARAMS)

if use_calibration:
    final_model = CalibratedClassifierCV(estimator=final_base, method="sigmoid", cv=5)
    final_model.fit(X_final_res, y_final_res)
else:
    final_model = final_base
    final_model.fit(X_final_res, y_final_res)

# Predict scoring probabilities
X_score_scaled = preprocessor.transform(X_score)
scoring_probs = final_model.predict_proba(X_score_scaled)[:, 1]

predictions_df = pd.DataFrame({
    "company_id": score_company_ids,
    "predicted_default_probability": scoring_probs,
    "risk_rating": [assign_risk_rating(p, optimal_threshold) for p in scoring_probs]
})

print("\nPredictions (first 10):")
print(predictions_df.head(10).to_string(index=False))
print(f"\nRisk distribution:\n{predictions_df['risk_rating'].value_counts().to_string()}")

predictions_df.to_csv(os.path.join(DATA_DIR, "predictions_raw.csv"), index=False)
print(f"\nSaved predictions_raw.csv ({predictions_df.shape[0]} rows)")


# ── 15. SHAP Explainability ───────────────────────────────────────────────────

print("\n=== Computing SHAP Values ===")

# SHAP TreeExplainer on the raw (uncalibrated) best tree model fitted on full data
if best_model_name in ("LightGBM", "Stacking Ensemble"):
    shap_base = lgb.LGBMClassifier(**LGBM_PARAMS)
elif best_model_name == "Logistic Regression":
    # LR has no TreeExplainer; fall back to XGBoost for feature importance
    shap_base = xgb.XGBClassifier(**XGB_PARAMS)
else:
    shap_base = xgb.XGBClassifier(**XGB_PARAMS)

shap_base.fit(X_scaled_all, y)
explainer = shap.TreeExplainer(shap_base)
shap_values = explainer(X_score_scaled)
shap_values.feature_names = feature_names

# Summary bar plot
shap.summary_plot(shap_values, max_display=15, show=False)
plt.title("SHAP Feature Importance (Scoring Set)")
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "shap_summary.png"), dpi=150)
plt.close()
print("Saved shap_summary.png")

shap_data = {
    "shap_values":            shap_values.values,
    "base_values":            shap_values.base_values,
    "feature_names":          feature_names,
    "scoring_features_scaled": X_score_scaled,
    "scoring_features_raw":   X_score.values,
}
with open(os.path.join(DATA_DIR, "shap_values.pkl"), "wb") as f:
    pickle.dump(shap_data, f)

print("Saved shap_values.pkl")


# ── 16. Final Summary ─────────────────────────────────────────────────────────

print("\n\n" + "="*70)
print("[DONE] MODELING COMPLETE")
print("="*70)
print(f"\nBest model       : {best_model_name}")
print(f"Brier Score Loss : {brier_cal:.4f}  (calibrated, OOF)")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"FNR              : {opt_fnr:.2%}  <- minimised (capital loss priority)")
print(f"FPR              : {opt_fpr:.2%}")
print(f"F1-Score         : {opt_f1:.2%}")
print(f"Precision        : {opt_prec:.2%}")
print(f"Recall (TPR)     : {opt_rec:.2%}")
print("="*70)
