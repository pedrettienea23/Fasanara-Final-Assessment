import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss

# Set Page Config
st.set_page_config(
    page_title="Fasanara Credit Risk Assessment Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #000000;
        color: #e0e0e0;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #121212;
        border: 1px solid #333333;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e0e0e0;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8a8a8a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Risk Rating Badges */
    .badge {
        padding: 5px 12px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(211, 47, 47, 0.15);
        color: #ff5252;
        border: 1px solid rgba(211, 47, 47, 0.4);
    }
    .badge-medium {
        background-color: rgba(245, 124, 0, 0.15);
        color: #ffb74d;
        border: 1px solid rgba(245, 124, 0, 0.4);
    }
    .badge-low {
        background-color: rgba(224, 224, 224, 0.15);
        color: #e0e0e0;
        border: 1px solid rgba(224, 224, 224, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid #222222;
    }
    
    /* Custom divider */
    hr {
        border-color: #222222 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADING ───────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "processed")
RAW_DATA_DIR = os.path.join(SCRIPT_DIR, "data")

@st.cache_data
def load_data():
    try:
        # Load raw scoring metadata
        scoring_raw = pd.read_csv(os.path.join(RAW_DATA_DIR, "scoring_companies.csv"))
        
        # Load predictions output
        predictions = pd.read_csv(os.path.join(DATA_DIR, "predictions_raw.csv"))
        
        # Merge them
        merged = pd.merge(predictions, scoring_raw, on="company_id")
        
        # Load SHAP pickle
        with open(os.path.join(DATA_DIR, "shap_values.pkl"), "rb") as f:
            shap_data = pickle.load(f)
            
        return merged, shap_data
    except Exception as e:
        st.error(f"Error loading data: {e}. Please run the preprocessing and modeling scripts first.")
        return None, None

df, shap_data = load_data()

# Check if data successfully loaded
if df is not None:
    
    # ─── SIDEBAR NAVIGATION ──────────────────────────────────────────────────
    
    st.sidebar.image("https://img.icons8.com/color/96/shield.png", width=80)
    st.sidebar.title("Fasanara Credit")
    st.sidebar.markdown("*AI Risk Assessment Suite*")
    st.sidebar.markdown("---")
    
    nav_option = st.sidebar.radio(
        "Navigation",
        ["Portfolio Overview", "Company Credit Memo", "Model Diagnostics"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Operating Target:**\n"
        "FNR ≈ 13.2% (Recall ~86.8%)\n\n"
        "**Decision Threshold:**\n"
        "p ≥ 0.0982 (High Risk)"
    )
    
    # ─── PORTFOLIO OVERVIEW ──────────────────────────────────────────────────
    
    if nav_option == "Portfolio Overview":
        st.title("💼 Portfolio Credit Risk Overview")
        st.markdown("Analytic summary of the 50 scored companies awaiting credit decisions.")
        st.markdown("---")
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        
        high_risk_count = (df["risk_rating"] == "High").sum()
        med_risk_count = (df["risk_rating"] == "Medium").sum()
        low_risk_count = (df["risk_rating"] == "Low").sum()
        avg_prob = df["predicted_default_probability"].mean()
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">50</div>
                <div class="metric-label">Total Scored Companies</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ef4444;">{high_risk_count}</div>
                <div class="metric-label">High Risk (Declined)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #f59e0b;">{med_risk_count}</div>
                <div class="metric-label">Medium Risk (Review)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #10b981;">{avg_prob:.2%}</div>
                <div class="metric-label">Average Default Probability</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### Scored Portfolio Distribution")
        
        # Distribution Charts
        chart_col1, chart_col2 = st.columns([1, 1])
        
        with chart_col1:
            # Risk Distribution Chart
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#000000')
            ax.set_facecolor('#121212')
            
            risk_counts = df["risk_rating"].value_counts().reindex(["Low", "Medium", "High"])
            colors = ["#b0bec5", "#f5a623", "#d32f2f"]
            
            bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor=(1.0, 1.0, 1.0, 0.1), width=0.5)
            ax.tick_params(colors='#e0e0e0')
            ax.spines['bottom'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel("Number of Companies", color="#e0e0e0")
            ax.set_title("Scoring Count by Risk Tier", color="#ffffff", fontsize=12)
            
            # Label bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', color='#e0e0e0', fontweight='bold')
                            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        with chart_col2:
            # Default Probability Boxplot
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor('#000000')
            ax.set_facecolor('#121212')
            
            sns.boxplot(x=df["predicted_default_probability"], color="#b0bec5", ax=ax)
            ax.axvline(x=0.0982, color='#d32f2f', linestyle='--', label="Threshold (0.0982)")
            ax.tick_params(colors='#e0e0e0')
            ax.spines['bottom'].set_color('#333333')
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel("Predicted Default Probability", color="#e0e0e0")
            ax.set_title("Spread of Default Probabilities", color="#ffffff", fontsize=12)
            ax.legend(facecolor='#121212', edgecolor='none', labelcolor='#e0e0e0')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        # Search & Filter Table
        st.markdown("### Portfolio Scoring Details")
        
        search_query = st.text_input("🔍 Search Company by Name or ID:")
        
        table_df = df[["company_id", "company_name", "sector", "country", "predicted_default_probability", "risk_rating"]].copy()
        table_df["predicted_default_probability"] = table_df["predicted_default_probability"].map(lambda p: f"{p:.2%}")
        
        if search_query:
            filtered_df = table_df[
                table_df["company_name"].str.contains(search_query, case=False) |
                table_df["company_id"].astype(str).str.contains(search_query)
            ]
        else:
            filtered_df = table_df
            
        st.dataframe(
            filtered_df,
            use_container_width=True,
            column_config={
                "company_id": "Company ID",
                "company_name": "Company Name",
                "sector": "Sector",
                "country": "Country",
                "predicted_default_probability": "Default Probability",
                "risk_rating": "Risk Rating"
            }
        )
        
    # ─── COMPANY CREDIT MEMO ─────────────────────────────────────────────────
    
    elif nav_option == "Company Credit Memo":
        st.title("📄 Individual Company Credit Assessment Memo")
        st.markdown("Generate a detailed financial summary, qualitative risk analysis, and local model explanation for any scored company.")
        st.markdown("---")
        
        # Selector
        company_selection = st.selectbox(
            "Select Company to Assess:",
            options=df["company_name"].tolist()
        )
        
        # Get selected row
        row = df[df["company_name"] == company_selection].iloc[0]
        company_id = row["company_id"]
        
        # Row layout
        header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
        
        with header_col1:
            st.markdown(f"## {row['company_name']}")
            st.markdown(f"**Sector:** {row['sector']} | **Country:** {row['country']} | **Company ID:** {company_id}")
            
        with header_col2:
            prob = row["predicted_default_probability"]
            st.markdown(f"""
            <div class="metric-card" style="padding: 10px;">
                <div class="metric-value">{prob:.2%}</div>
                <div class="metric-label">Default Probability</div>
            </div>
            """, unsafe_allow_html=True)
            
        with header_col3:
            rating = row["risk_rating"]
            if rating == "High":
                badge_html = '<span class="badge badge-high" style="font-size:1.5rem; padding: 10px 20px;">DECLINE (High Risk)</span>'
            elif rating == "Medium":
                badge_html = '<span class="badge badge-medium" style="font-size:1.5rem; padding: 10px 20px;">REFER (Medium Risk)</span>'
            else:
                badge_html = '<span class="badge badge-low" style="font-size:1.5rem; padding: 10px 20px;">APPROVE (Low Risk)</span>'
                
            st.markdown(f"""
            <div style="text-align: center; margin-top: 10px;">
                {badge_html}
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Tabs for details
        memo_tab1, memo_tab2 = st.tabs(["📊 Financial & Qualitative Profile", "🧠 Local Model Interpretability (SHAP)"])
        
        with memo_tab1:
            profile_col1, profile_col2 = st.columns(2)
            
            with profile_col1:
                st.markdown("### Financial Indicators Profile")
                
                fin_data = {
                    "Metric": [
                        "Revenue", 
                        "EBITDA Margin", 
                        "Debt Ratio", 
                        "Interest Coverage", 
                        "Cash Ratio", 
                        "Years in Operation", 
                        "Employee Count", 
                        "Revenue Growth"
                    ],
                    "Value": [
                        f"€{row['revenue_m']:.2f}M",
                        f"{row['ebitda_margin']:.2%}",
                        f"{row['debt_ratio']:.2%}",
                        f"{row['interest_coverage']:.2f}x",
                        f"{row['cash_ratio']:.2%}",
                        f"{int(row['years_in_operation'])} years",
                        f"{int(row['employee_count'])}",
                        f"{row['revenue_growth']:.2%}"
                    ]
                }
                st.table(pd.DataFrame(fin_data))
                
            with profile_col2:
                st.markdown("### Text-Based Qualitative Signals")
                st.markdown("*Signals extracted from business narrative descriptions.*")
                
                # Extract risks and mitigants
                desc = row["business_description"]
                st.markdown(f"**Raw Description:** *\"{desc}\"*")
                
                risk_keywords = [
                    ("dependence on a few key accounts", "Key Account Dependence"),
                    ("exposed to volatile commodity prices", "Volatile Commodity Prices"),
                    ("short operating history", "Short Operating History"),
                    ("regulatory uncertainty", "Regulatory Uncertainty"),
                    ("high customer concentration", "High Customer Concentration"),
                    ("margin compression", "Margin Compression"),
                    ("declining revenue", "Declining Revenue"),
                    ("legacy debt burden", "Legacy Debt Burden"),
                    ("weak liquidity position", "Weak Liquidity Position"),
                    ("tight cash position", "Tight Cash Position")
                ]
                
                strength_keywords = [
                    ("recurring subscription revenue", "Recurring Subscription Revenue"),
                    ("long-term government contracts", "Government Contracts"),
                    ("strong market position", "Strong Market Position"),
                    ("resilient demand profile", "Resilient Demand Profile"),
                    ("diversified customer base", "Diversified Customer Base"),
                    ("stable supplier relationships", "Stable Suppliers"),
                    ("high-quality management team", "High-Quality Management")
                ]
                
                found_risks = [name for phrase, name in risk_keywords if phrase in desc.lower()]
                found_strengths = [name for phrase, name in strength_keywords if phrase in desc.lower()]
                
                st.markdown("#### Detected Risk Indicators")
                if found_risks:
                    for r in found_risks:
                        st.markdown(f"🔴 **{r}**")
                else:
                    st.markdown("🟢 No qualitative risk indicators detected.")
                    
                st.markdown("#### Detected Strength Indicators")
                if found_strengths:
                    for s in found_strengths:
                        st.markdown(f"🔵 **{s}**")
                else:
                    st.markdown("🟡 No qualitative strength indicators detected.")
                    
            st.markdown("---")
            st.markdown("### 📝 Structured Credit Memo (Analyst Recommendation)")
            
            # Recommendation memo text
            decision = "DECLINE" if rating == "High" else ("REFER FOR MANUAL AUDIT" if rating == "Medium" else "APPROVE")
            recom_style = "color:#ef4444;" if rating == "High" else ("color:#f59e0b;" if rating == "Medium" else "color:#10b981;")
            
            st.markdown(f"""
            ```text
            FASANARA CREDIT ASSESSMENT MEMO
            ==================================================
            COMPANY NAME: {row['company_name']}
            COMPANY ID  : {company_id}
            DATE        : 2026-06-10
            
            1. EXECUTIVE RECOMMENDATION
            ---------------------------
            PROPOSAL    : {decision}
            MODEL RATING: {rating} Risk
            EST. DEFAULT PROBABILITY: {prob:.2%} (Decision threshold: 9.82%)
            
            2. KEY FINANCIAL FINDINGS
            -------------------------
            * Revenue stands at €{row['revenue_m']:.2f}M, with operating (EBITDA) margin of {row['ebitda_margin']:.2%}.
            * Leverage (Debt Ratio) is {row['debt_ratio']:.2%}, backed by interest coverage of {row['interest_coverage']:.2f}x.
            * Years in operation: {int(row['years_in_operation'])} years.
            
            3. RISK & MITIGATING FACTORS SUMMARY
            -------------------------------------
            * Qualitative Risks: {", ".join(found_risks) if found_risks else "None detected"}
            * Qualitative Mitigants: {", ".join(found_strengths) if found_strengths else "None detected"}
            
            4. CONCLUSION
            -------------
            {row['company_name']} exhibits a default risk that is {'above' if rating in ("High", "Medium") else 'well within'} acceptable thresholds. 
            {"The company represents high default risk; capital reservation mandates a DECLINE of the credit facility." if rating == "High" else ("The default risk is borderline. MANUAL REVIEW is required to evaluate mitigants." if rating == "Medium" else "The credit request is approved subject to standard covenants.")}
            ==================================================
            ```
            """, unsafe_allow_html=True)
            
        with memo_tab2:
            st.markdown("### Why did the model predict this risk score?")
            st.markdown("The SHAP values explain how much each financial or text indicator pushed the predicted probability up (towards default, red) or down (towards safety, blue).")
            
            # Retrieve SHAP indices
            scoring_idx = df[df["company_id"] == company_id].index[0]
            
            # Feature name mapping
            feature_display_names = {
                "revenue_m": "Revenue (€M)",
                "ebitda_margin": "EBITDA Margin (%)",
                "debt_ratio": "Debt Ratio (%)",
                "interest_coverage": "Interest Coverage (x)",
                "cash_ratio": "Cash Ratio (%)",
                "years_in_operation": "Years in Operation",
                "employee_count": "Headcount",
                "revenue_growth": "Revenue Growth (%)",
                "distress_score": "Financial Distress Score",
                "debt_cov_profitability": "Debt to Profitability",
                "debt_to_earnings": "Debt to Earnings",
                "debt_to_coverage": "Debt Service Coverage",
                "net_leverage": "Net Leverage",
                "debt_margin_stress": "Debt Margin Interaction Stress",
                "liquidity_coverage": "Liquidity Coverage Buffer",
                "flag_risk_dependence_on_a_few_key_accounts": "Risk: Key Account Dependence",
                "flag_risk_exposed_to_volatile_commodity_prices": "Risk: Volatile Commodity Prices",
                "flag_risk_short_operating_history": "Risk: Short Operating History",
                "flag_risk_regulatory_uncertainty": "Risk: Regulatory Uncertainty",
                "flag_risk_high_customer_concentration": "Risk: High Customer Concentration",
                "flag_risk_margin_compression": "Risk: Margin Compression",
                "flag_risk_declining_revenue": "Risk: Declining Revenue",
                "flag_risk_legacy_debt_burden": "Risk: Legacy Debt Burden",
                "flag_risk_weak_liquidity_position": "Risk: Weak Liquidity Position",
                "flag_risk_tight_cash_position": "Risk: Tight Cash Position",
                "flag_strength_recurring_subscription_revenue": "Strength: Recurring Subscription Revenue",
                "flag_strength_long_term_government_contracts": "Strength: Government Contracts",
                "flag_strength_strong_market_position": "Strength: Strong Market Position",
                "flag_strength_resilient_demand_profile": "Strength: Resilient Demand Profile",
                "flag_strength_diversified_customer_base": "Strength: Diversified Customer Base",
                "flag_strength_stable_supplier_relationships": "Strength: Stable Suppliers",
                "flag_strength_high_quality_management_team": "Strength: High-Quality Management"
            }
            
            # Reconstruct names from shap data
            shaps = shap_data["shap_values"][scoring_idx]
            feat_names = shap_data["feature_names"]
            raw_vals = shap_data["scoring_features_raw"][scoring_idx]
            
            # Sort by absolute contribution
            sorted_indices = np.argsort(np.abs(shaps))[::-1][:10]
            
            plot_names = []
            plot_vals = []
            plot_raw_text = []
            
            for idx in sorted_indices:
                name = feat_names[idx]
                clean = feature_display_names.get(name, name.replace("_", " ").title())
                
                # Format raw feature values nicely
                raw_v = raw_vals[idx]
                if name == "revenue_m":
                    raw_str = f"€{raw_v:.1f}M"
                elif name in ["ebitda_margin", "debt_ratio", "cash_ratio", "revenue_growth"]:
                    raw_str = f"{raw_v:.1%}"
                elif name == "interest_coverage":
                    raw_str = f"{raw_v:.2f}x"
                elif name in ["years_in_operation", "employee_count"]:
                    raw_str = f"{int(raw_v)}"
                elif name.startswith("flag_") or name.startswith("sector_") or name.startswith("country_"):
                    raw_str = "Present" if raw_v == 1 else "Absent"
                else:
                    raw_str = f"{raw_v:.2f}"
                    
                plot_names.append(f"{clean} ({raw_str})")
                plot_vals.append(shaps[idx])
                
            # Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#000000')
            ax.set_facecolor('#121212')
            
            colors = ['#d32f2f' if v > 0 else '#b0bec5' for v in plot_vals]
            
            bars = ax.barh(plot_names[::-1], plot_vals[::-1], color=colors[::-1], height=0.5, edgecolor=(1.0, 1.0, 1.0, 0.05))
            ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
            ax.tick_params(colors='#e0e0e0', labelsize=9)
            ax.spines['bottom'].set_color('#333333')
            ax.spines['left'].set_color('#333333')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('SHAP Value (Contribution to Log-Odds of Default)', color="#e0e0e0", fontsize=9)
            ax.set_title('Top 10 Decision Drivers (SHAP Explanation)', color="#ffffff", fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.markdown(
                "💡 **How to interpret:** "
                "Red bars indicate factors that **increased** the default probability (pushed risk higher). "
                "Blue bars indicate protective factors that **decreased** default risk."
            )
            
    # ─── MODEL DIAGNOSTICS ───────────────────────────────────────────────────
    
    elif nav_option == "Model Diagnostics":
        st.title("🛡️ Credit Model Diagnostics & Performance")
        st.markdown("Detailed breakdown of model development metrics, probability calibration, and feature importances.")
        st.markdown("---")
        
        diag_col1, diag_col2 = st.columns([1, 1])
        
        with diag_col1:
            st.markdown("### 📈 Model Comparison (Stratified 5-Fold CV)")
            
            comparison_data = {
                "Model": ["Logistic Regression", "Tuned XGBoost", "Calibrated XGBoost (Platt)"],
                "ROC-AUC": [0.7646, 0.7831, 0.7880],
                "PR-AUC": [0.4702, 0.4927, 0.4925],
                "Brier Score": [0.1862, 0.1570, 0.1192],
                "FNR (Operating Point)": ["34.97%", "36.74%", "13.22%"],
                "FPR (Operating Point)": ["25.91%", "20.47%", "49.76%"],
                "Threshold": [0.5000, 0.5000, 0.0982]
            }
            
            st.table(pd.DataFrame(comparison_data))
            
            st.markdown(
                "**Why Calibrated XGBoost is selected:**\n"
                "* **Probability Quality**: Brier Score dropped from 0.1570 (raw) to 0.1192 (calibrated), meaning default probabilities are highly reliable for loan pricing.\n"
                "* **Capital Protection**: By tuning the decision threshold to `0.0982`, we successfully limited False Negatives (actual defaults classified as safe) to **13.22%** (Recall/TPR ~86.8%)."
            )
            
        with diag_col2:
            st.markdown("### 📊 Probability Calibration Reliability")
            st.image(os.path.join(DATA_DIR, "calibration_curve.png"), use_container_width=True)
            st.markdown("*Sigmoid Platt Calibration successfully maps output scores to real default frequencies, bringing the curves closest to the perfectly calibrated diagonal.*")
            
        st.markdown("---")
        st.markdown("### 🧬 Global Feature Importance (Top SHAP Contributors across Scoring Set)")
        st.image(os.path.join(DATA_DIR, "shap_summary.png"), use_container_width=True)
        st.markdown("*The financial distress score (distress_score) and debt-to-earnings ratios carry the highest weight in classifying corporate defaults.*")
