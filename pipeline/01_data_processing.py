#!/usr/bin/env python
# coding: utf-8

# # AI Credit Risk Assessment - Data Processing & Feature Engineering
# 
# This notebook builds the data processing and feature engineering pipeline. It performs the following steps:
# 1. Loads and joins raw datasets.
# 2. Extracts binary text features from qualitative business descriptions using exact string matching.
# 3. Computes domain-specific interaction ratios.
# 4. Performs one-hot encoding for categorical variables while ensuring train/scoring schemas are aligned.
# 5. Saves the final processed datasets.
# 

# In[1]:


import pandas as pd
import numpy as np
import re
import os

print("Imports completed successfully.")


# In[2]:


# Define paths
DATA_DIR = "../data"
PROCESSED_DIR = "../data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load training datasets
train_companies = pd.read_csv(os.path.join(DATA_DIR, "train_companies.csv"))
train_narratives = pd.read_csv(os.path.join(DATA_DIR, "train_narratives.csv"))
train_outcomes = pd.read_csv(os.path.join(DATA_DIR, "train_outcomes.csv"))

# Load scoring dataset
scoring_companies = pd.read_csv(os.path.join(DATA_DIR, "scoring_companies.csv"))

print(f"train_companies shape: {train_companies.shape}")
print(f"train_narratives shape: {train_narratives.shape}")
print(f"train_outcomes shape: {train_outcomes.shape}")
print(f"scoring_companies shape: {scoring_companies.shape}")


# In[3]:


# Merge training files
df_train = train_companies.merge(train_outcomes, on="company_id").merge(train_narratives, on="company_id")
print(f"Merged training dataset shape: {df_train.shape}")
df_train.head()


# In[4]:


# Exact textual phrases verified in the dataset
risks = [
    "declining revenue", 
    "dependence on a few key accounts", 
    "exposed to volatile commodity prices", 
    "high customer concentration", 
    "legacy debt burden", 
    "margin compression", 
    "regulatory uncertainty", 
    "short operating history", 
    "tight cash position", 
    "weak liquidity position"
]

strengths = [
    "diversified customer base", 
    "high-quality management team", 
    "long-term government contracts", 
    "recurring subscription revenue", 
    "resilient demand profile", 
    "stable supplier relationships", 
    "strong market position"
]

def extract_nlp_features(df, text_col):
    # Create copy to avoid SettingWithCopyWarning
    df = df.copy()

    # 1. Binary flags for each risk phrase
    for r in risks:
        col_name = f"flag_risk_{r.lower().replace(' ', '_').replace('-', '_')}"
        df[col_name] = df[text_col].str.contains(r, case=False, regex=False).astype(int)

    # 2. Binary flags for each strength phrase
    for s in strengths:
        col_name = f"flag_strength_{s.lower().replace(' ', '_').replace('-', '_')}"
        df[col_name] = df[text_col].str.contains(s, case=False, regex=False).astype(int)

    # 3. Aggregate counts
    risk_cols = [f"flag_risk_{r.lower().replace(' ', '_').replace('-', '_')}" for r in risks]
    strength_cols = [f"flag_strength_{s.lower().replace(' ', '_').replace('-', '_')}" for s in strengths]

    df["n_risk"] = df[risk_cols].sum(axis=1)
    df["n_strength"] = df[strength_cols].sum(axis=1)

    return df

# Apply to training and scoring datasets
df_train = extract_nlp_features(df_train, "business_description")
df_score = extract_nlp_features(scoring_companies, "business_description")

print("NLP Feature extraction complete.")
print(f"df_train columns: {df_train.shape[1]}, df_score columns: {df_score.shape[1]}")


# In[5]:


def engineer_financial_interactions(df):
    df = df.copy()

    # Composite distress score (highest correlation in training data)
    df["distress_score"] = (df["debt_ratio"] / df["interest_coverage"]) - df["ebitda_margin"] - df["cash_ratio"]

    # Leverage normalized by coverage and profitability
    df["debt_cov_profitability"] = df["debt_ratio"] / (df["interest_coverage"] * df["ebitda_margin"])

    # Debt relative to earnings power
    df["debt_to_earnings"] = df["debt_ratio"] / df["ebitda_margin"]

    # Direct debt service stress
    df["debt_to_coverage"] = df["debt_ratio"] / df["interest_coverage"]

    # Net leverage (gross debt minus liquid buffer)
    df["net_leverage"] = df["debt_ratio"] - df["cash_ratio"]

    # Interaction: high debt and low margins
    df["debt_margin_stress"] = df["debt_ratio"] * (1 - df["ebitda_margin"])

    # Liquidity coverage (protective cash buffer interaction)
    df["liquidity_coverage"] = df["cash_ratio"] * df["interest_coverage"]

    return df

# Apply to training and scoring datasets
df_train = engineer_financial_interactions(df_train)
df_score = engineer_financial_interactions(df_score)

print("Financial interactions engineered.")
print(f"df_train columns: {df_train.shape[1]}, df_score columns: {df_score.shape[1]}")


# In[6]:


# Drop irrelevant columns (company_name and raw business_description) to avoid leakage/schema mismatch
drop_cols = ["company_name", "business_description"]
df_train_dropped = df_train.drop(columns=drop_cols)
df_score_dropped = df_score.drop(columns=drop_cols)

# Separate target
y_train = df_train_dropped["defaulted"]
X_train_raw = df_train_dropped.drop(columns=["defaulted"])
X_score_raw = df_score_dropped.copy()

# Perform one-hot encoding on sector and country
X_train_encoded = pd.get_dummies(X_train_raw, columns=["sector", "country"], drop_first=True, dtype=int)
X_score_encoded = pd.get_dummies(X_score_raw, columns=["sector", "country"], drop_first=True, dtype=int)

# Align columns to make sure training and scoring features are identical
# Get missing columns in scoring set
missing_cols = set(X_train_encoded.columns) - set(X_score_encoded.columns)
for col in missing_cols:
    X_score_encoded[col] = 0

# Sort columns so they are in the exact same order
X_score_encoded = X_score_encoded[X_train_encoded.columns]

# Construct final processed dataframes
train_processed = X_train_encoded.copy()
train_processed["defaulted"] = y_train
scoring_processed = X_score_encoded.copy()

print(f"Processed training set shape: {train_processed.shape}")
print(f"Processed scoring set shape: {scoring_processed.shape}")
assert list(train_processed.drop(columns=['defaulted']).columns) == list(scoring_processed.columns), "Columns do not align!"


# In[7]:


# Save datasets
train_processed.to_csv(os.path.join(PROCESSED_DIR, "train_processed.csv"), index=False)
scoring_processed.to_csv(os.path.join(PROCESSED_DIR, "scoring_processed.csv"), index=False)

print("Processed files saved successfully to:", PROCESSED_DIR)

