# Data Description

This folder contains a fully synthetic dataset designed for the Fasanara AI Credit Risk Analyst Challenge. The files are fictional and intended for model prototyping, feature engineering, and explanation generation.

## Files in the data folder

### 1. train_companies.csv
This file contains structured company-level features for the labelled training set.

Key columns:
- `company_id`: unique identifier for each company.
- `company_name`: fictional company name used for readability only.
- `sector`: industry segment such as Retail, Manufacturing, Software, Healthcare, Logistics, Renewable Energy, Real Estate, Hospitality, Crypto / Digital Assets, or Consumer Finance.
- `country`: country of operation among the synthetic market set.
- `revenue_m`: annual revenue in millions.
- `ebitda_margin`: operating profitability as a decimal fraction.
- `debt_ratio`: total debt relative to financial capacity.
- `interest_coverage`: ability to service interest expenses.
- `cash_ratio`: short-term liquidity measure.
- `years_in_operation`: number of years the company has been active.
- `employee_count`: approximate employee headcount.
- `revenue_growth`: year-over-year revenue growth rate.

### 2. train_narratives.csv
This file contains qualitative narrative text for each training company.

Key columns:
- `company_id`: links back to `train_companies.csv`.
- `business_description`: free-text description of the company, including qualitative signals such as declining revenue, high customer concentration, regulatory uncertainty, weak liquidity, strong market position, short operating history, and recurring subscription revenue.

### 3. train_outcomes.csv
This file contains the target label used for supervised learning.

Key columns:
- `company_id`: links back to the training companies.
- `defaulted`: binary outcome where `1` means the company defaulted and `0` means it did not.

### 4. scoring_companies.csv
This file contains unseen companies that candidates must score.

Key columns:
- `company_id`, `company_name`, `sector`, `country`, `revenue_m`, `ebitda_margin`, `debt_ratio`, `interest_coverage`, `cash_ratio`, `years_in_operation`, `employee_count`, `revenue_growth`
- `business_description`: narrative text for the company to be scored.

## How the data is intended to be used
1. Join `train_companies.csv` with `train_narratives.csv` using `company_id`.
2. Join the result with `train_outcomes.csv` to create a supervised training dataset.
3. Use the combined features to train a baseline risk model.
4. Apply the same feature engineering logic to `scoring_companies.csv` to generate default probabilities and analyst-style explanations.

## Notes for candidates
- The dataset is synthetic, so it should be treated as a realistic prototype challenge rather than real-world credit data.
- The narrative text is intentionally varied and may contain both positive and negative risk signals.
- The target label is generated from a hidden synthetic risk function, so the relationship between features and default risk is real but not perfectly obvious.
- The scoring companies do not contain the `defaulted` label and are meant to be predicted.
