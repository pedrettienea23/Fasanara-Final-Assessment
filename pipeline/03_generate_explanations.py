import os
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Set to True to generate explanations via LLM APIs or local templates
GENERATE_EXPLANATIONS = False

# Define paths
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
predictions_raw = pd.read_csv("data/processed/predictions_raw.csv")
scoring_companies = pd.read_csv("data/scoring_companies.csv")

# Load SHAP data
with open("data/processed/shap_values.pkl", "rb") as f:
    shap_data = pickle.load(f)

shap_values = shap_data["shap_values"]
base_values = shap_data["base_values"]
feature_names = shap_data["feature_names"]
scoring_features_raw = shap_data["scoring_features_raw"]

print("Loaded all inputs. Scoring predictions shape:", predictions_raw.shape)

# Helper function to map feature names to clean readable descriptions
feature_description_map = {
    "revenue_m": "revenue of €{:.1f}M",
    "ebitda_margin": "operating margin (EBITDA) of {:.1%}",
    "debt_ratio": "debt ratio of {:.1%}",
    "interest_coverage": "interest coverage of {:.1f}x",
    "cash_ratio": "cash ratio of {:.1%}",
    "years_in_operation": "operating history of {:d} years",
    "employee_count": "headcount of {:d} employees",
    "revenue_growth": "revenue growth of {:.1%}",
    "distress_score": "composite distress score",
    "debt_cov_profitability": "debt-to-earnings/coverage ratio",
    "debt_to_earnings": "debt-to-earnings ratio",
    "debt_to_coverage": "debt service coverage ratio",
    "net_leverage": "net leverage (debt minus cash)",
    "debt_margin_stress": "debt margin interaction stress",
    "liquidity_coverage": "cash-interest coverage buffer",
    "n_risk": "aggregate narrative risks count",
    "n_strength": "aggregate narrative strengths count"
}

def clean_feature_value(name, val):
    # Translate binary flags and one-hot variables nicely
    if name.startswith("flag_risk_"):
        phrase = name.replace("flag_risk_", "").replace("_", " ")
        return f"risk signal: '{phrase}'"
    elif name.startswith("flag_strength_"):
        phrase = name.replace("flag_strength_", "").replace("_", " ")
        return f"mitigant: '{phrase}'"
    elif name.startswith("sector_"):
        sector = name.replace("sector_", "")
        return f"operation in sector '{sector}'"
    elif name.startswith("country_"):
        country = name.replace("country_", "")
        return f"operation in country '{country}'"
    
    desc_template = feature_description_map.get(name, name + ": {:.2f}")
    try:
        # Format the value nicely depending on template
        if "%" in desc_template:
            return desc_template.format(val)
        elif "d" in desc_template:
            return desc_template.format(int(val))
        elif ".1f" in desc_template:
            return desc_template.format(val)
        else:
            return desc_template.format(val)
    except Exception:
        return f"{name} of {val:.2f}"

def get_top_shap_drivers(company_idx, top_n=3):
    # Extract feature values and SHAP values for this company
    shaps = shap_values[company_idx]
    raw_vals = scoring_features_raw[company_idx]
    
    # Sort features by SHAP value
    sorted_indices = np.argsort(shaps)
    
    # Positive contributors (increasing risk)
    pos_drivers = []
    # Negative contributors (decreasing risk / mitigants)
    neg_drivers = []
    
    # Extract positive contributors (highest SHAP values)
    for idx in sorted_indices[::-1]:
        if shaps[idx] > 0.01: # Only include meaningful contributions
            name = feature_names[idx]
            val = raw_vals[idx]
            pos_drivers.append((name, val, shaps[idx]))
        if len(pos_drivers) >= top_n:
            break
            
    # Extract negative contributors (lowest SHAP values)
    for idx in sorted_indices:
        if shaps[idx] < -0.01: # Only include meaningful contributions
            name = feature_names[idx]
            val = raw_vals[idx]
            neg_drivers.append((name, val, shaps[idx]))
        if len(neg_drivers) >= top_n:
            break
            
    return pos_drivers, neg_drivers

def generate_local_explanation(company_name, prob, risk_rating, pos_drivers, neg_drivers, desc):
    # Construct a template-based explanation from the actual SHAP values
    pos_strs = [clean_feature_value(name, val) for name, val, _ in pos_drivers]
    neg_strs = [clean_feature_value(name, val) for name, val, _ in neg_drivers]
    
    explanation = f"Company '{company_name}' is rated as {risk_rating} Risk with a default probability of {prob:.1%}. "
    
    if pos_strs:
        explanation += f"The default probability is primarily driven by {', '.join(pos_strs)}. "
    else:
        explanation += "The default probability is low and stable across base features. "
        
    if neg_strs:
        explanation += f"These risks are partially mitigated by positive indicators: {', '.join(neg_strs)}."
    else:
        explanation += "There are no significant mitigating factors detected in the metrics."
        
    return explanation

def generate_llm_explanation(company_name, prob, risk_rating, pos_drivers, neg_drivers, desc):
    # Build prompt context
    pos_desc = ", ".join([f"{name} ({clean_feature_value(name, val)}) contributing +{shap:.3f} to log-odds" for name, val, shap in pos_drivers])
    neg_desc = ", ".join([f"{name} ({clean_feature_value(name, val)}) contributing {shap:.3f} to log-odds" for name, val, shap in neg_drivers])
    
    system_prompt = (
        "You are an expert Credit Risk Analyst at Fasanara. Generate a professional, concise, "
        "and data-grounded credit assessment summary (1-2 sentences) for a company. "
        "Use ONLY the facts provided. Speak directly to the primary drivers and mitigating factors. "
        "Do not invent any information."
    )
    
    prompt = f"""
Company Name: {company_name}
Model Prediction: {prob:.1%} default probability (Risk Rating: {risk_rating})
Raw Business Narrative: {desc}

Top Risk Drivers (increasing default probability):
{pos_desc if pos_desc else "None"}

Mitigating Factors (decreasing default probability):
{neg_desc if neg_desc else "None"}

Generate the analyst explanation:
"""
    # Call OpenAI if key is present
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            
    # Call Anthropic if key is present
    elif os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from anthropic import Anthropic
            client = Anthropic()
            response = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=150,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Error calling Anthropic API: {e}")
            
    # Fallback to local template-based narrative
    return generate_local_explanation(company_name, prob, risk_rating, pos_drivers, neg_drivers, desc)

# Process all companies
final_rows = []

print("Generating analyst-style explanations...")
for idx, row in predictions_raw.iterrows():
    company_id = int(row["company_id"])
    prob = float(row["predicted_default_probability"])
    risk_rating = str(row["risk_rating"])
    
    # Get original metadata and narrative
    orig_row = scoring_companies[scoring_companies["company_id"] == company_id].iloc[0]
    company_name = orig_row["company_name"]
    desc = orig_row["business_description"]
    
    # Find the corresponding index in the scoring set
    scoring_idx = scoring_companies[scoring_companies["company_id"] == company_id].index[0]
    
    # Get top drivers based on SHAP values
    pos_drivers, neg_drivers = get_top_shap_drivers(scoring_idx)
    
    # Generate explanation
    if GENERATE_EXPLANATIONS:
        explanation = generate_llm_explanation(company_name, prob, risk_rating, pos_drivers, neg_drivers, desc)
    else:
        explanation = ""
    
    final_rows.append({
        "company_id": company_id,
        "predicted_default_probability": prob,
        "risk_rating": risk_rating,
        "explanation": explanation
    })

# Save final predictions
df_final = pd.DataFrame(final_rows)
output_path = os.path.join(OUTPUT_DIR, "predictions.csv")
df_final.to_csv(output_path, index=False)

print(f"Explanations generation finished. Saved to {output_path}")
print(df_final.head(5))
