# AI Credit Risk Analyst Challenge

## Business Context
Fasanara is building a proof-of-concept AI-assisted credit risk assessment tool for SME and company borrowers. Candidates are asked to combine structured financial data, unstructured company narratives, machine learning, and AI-style explanations to build a practical prototype for credit risk screening.

## Objective
The goal is to build a prototype that:

1. Trains a model to estimate the probability of default.
2. Uses both structured company data and narrative descriptions where possible.
3. Produces default predictions for unseen scoring companies.
4. Generates analyst-style explanations for each prediction.
5. Optionally provides a simple interface such as a script, notebook, or Streamlit app.

## Time Limit
Candidates have **3 hours** to build and present their solution.

## Data Description
The repository contains the following synthetic datasets:

- `data/train_companies.csv`: structured financial and company attributes for labelled training companies.
- `data/train_narratives.csv`: business descriptions and qualitative risk notes for training companies.
- `data/train_outcomes.csv`: binary default labels for the training companies.
- `data/scoring_companies.csv`: unseen companies that require prediction and explanation.

## Required Deliverables
By the end of the challenge, candidates should submit:

1. Working code.
2. A short README or notes explaining their approach.
3. A predictions file saved under `outputs/predictions.csv`.
4. A short presentation/demo to the panel.

The expected `outputs/predictions.csv` format is:

```csv
company_id,predicted_default_probability,risk_rating,explanation
```

Where:

- `predicted_default_probability` is a number between 0 and 1.
- `risk_rating` can be Low, Medium, High, or similar.
- `explanation` is a short analyst-style explanation.

## Minimum Requirements
Candidates should at least:

- Load and join the provided datasets.
- Train a baseline predictive model.
- Generate predictions for the scoring companies.
- Provide explanations for those predictions.
- Commit their work to their forked repository.

## Bonus Ideas
Optional enhancements include:

- Using text features from business descriptions.
- Adding embeddings or richer NLP features.
- Using an LLM or local prompt-based method to generate explanations.
- Building a Streamlit app.
- Adding model evaluation.
- Adding feature importance or explainability.
- Improving project structure and reusable functions.
- Adding input validation and error handling.

## AI Tool Policy
Candidates are encouraged to use AI-assisted coding tools such as Cursor, Windsurf, ChatGPT, Claude, Gemini, GitHub Copilot, or similar. They should be prepared to explain:

- What they built.
- How AI tools were used.
- Key design decisions.
- Limitations and next steps.

## Submission Instructions
Candidates should:

1. Fork the repository.
2. Work on their fork.
3. Commit their solution.
4. Ensure `outputs/predictions.csv` exists.
5. Share the forked repository link before the deadline.
