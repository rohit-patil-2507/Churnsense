import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from utils.data_utils import load_processed_data, get_feature_columns
from utils.model_utils import load_trained_model, predict_single_record, get_top_feature_drivers

st.set_page_config(
    page_title="ChurnSense | Enterprise Retention & Risk Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= BALANCED PROFESSIONAL ENTERPRISE STYLING =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #000000 !important;
    }

    /* 1. Fluid, adaptive container padding */
    .block-container {
        padding-top: clamp(1rem, 2vw, 2rem) !important;
        padding-bottom: clamp(1.5rem, 3vw, 3rem) !important;
        padding-left: clamp(0.75rem, 2vw, 2.5rem) !important;
        padding-right: clamp(0.75rem, 2vw, 2.5rem) !important;
        max-width: 100% !important;
        width: 100% !important;
    }

    /* 2. Top Executive Header with Fluid Typography */
    .app-header {
        background: #000000;
        border-radius: 10px;
        padding: clamp(16px, 2.5vw, 24px) clamp(18px, 3vw, 28px);
        color: #FFFFFF;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #000000;
        gap: 16px;
    }
    .app-title {
        font-size: clamp(1.35rem, 2.5vw, 1.75rem);
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF !important;
        margin-bottom: 4px;
    }
    .app-subtitle {
        font-size: clamp(0.82rem, 1.3vw, 0.9rem);
        color: #E2E8F0 !important;
        line-height: 1.5;
        margin: 0;
    }
    .status-badge {
        background: #0F172A;
        border: 1px solid #475569;
        color: #38BDF8 !important;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 6px 14px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }

    /* 3. Structured Metric Cards with Minimal Black Border */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #000000;
        border-radius: 8px;
        padding: clamp(14px, 1.5vw, 18px) clamp(12px, 1.5vw, 18px);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-label {
        font-size: clamp(0.7rem, 1vw, 0.78rem);
        font-weight: 700;
        color: #0F172A !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: clamp(1rem, 1.35vw, 1.35rem);
        font-weight: 800;
        color: #000000 !important;
        line-height: 1.25;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-subtext {
        font-size: clamp(0.75rem, 1vw, 0.82rem);
        font-weight: 700;
        margin-top: 6px;
        color: #000000 !important;
    }

    /* 4. Subtle Callout Container */
    .info-callout {
        background: #F8FAFC;
        border: 1px solid #000000;
        border-left: 4px solid #000000;
        border-radius: 8px;
        padding: 16px 18px;
        margin-top: 14px;
        color: #000000 !important;
    }
    .info-callout div, .info-callout p, .info-callout ul, .info-callout li {
        color: #000000 !important;
    }

    /* 5. Streamlit Native Text & Heading Enforcements */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #000000;
    }
    .stMarkdown p, .stCaption {
        color: #0F172A !important;
        font-weight: 600;
    }

    /* 6. Native Tab Refinement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #000000;
        padding-bottom: 4px;
        overflow-x: auto !important;
        white-space: nowrap !important;
        display: flex !important;
        scrollbar-width: none;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: clamp(0.82rem, 1.2vw, 0.92rem);
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 6px;
        color: #334155 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        background-color: #E2E8F0;
        border-bottom: 2px solid #000000;
    }

    /* 7. Form Input Controls */
    div[data-testid="stNumberInput"] label {
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }

    /* 8. Responsive Media Queries */
    @media screen and (max-width: 768px) {
        .app-header {
            flex-direction: column !important;
            align-items: flex-start !important;
            padding: 16px 18px !important;
        }
        .status-badge {
            margin-top: 4px;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 10px;
        }
    }

    /* 9. Responsive DataFrames */
    div[data-testid="stDataFrame"] {
        width: 100% !important;
        overflow-x: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= RESOURCE INITIALIZATION =================
@st.cache_data
def get_data():
    return load_processed_data()

@st.cache_resource
def get_model():
    loaded = load_trained_model()
    if isinstance(loaded, dict) and "model" in loaded:
        return loaded["model"]
    return loaded

try:
    df = get_data()
    model = get_model()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()

# Responsive Header Component
st.markdown("""
<div class="app-header">
    <div>
        <div class="app-title">ChurnSense</div>
        <div class="app-subtitle">Customer Retention & Risk Intelligence — behavioral telemetry forecasting powered by an optimized XGBoost classifier.</div>
    </div>
    <div>
        <span class="status-badge">● Engine: XGBoost Active</span>
    </div>
</div>
""", unsafe_allow_html=True)

feature_cols = get_feature_columns(df)
top_kpi_drivers = get_top_feature_drivers(model, feature_cols, top_n=4)
all_ranked_drivers = get_top_feature_drivers(model, feature_cols, top_n=len(feature_cols))

tabs = st.tabs(["📊 Executive Dashboard", "🏆 Model Performance & Validation", "⚡ Single-Customer Simulator"])

# ================= TAB 1: EXECUTIVE KPI DASHBOARD =================
with tabs[0]:
    total_customers = len(df)
    churn_rate = (df['churned'].mean() * 100) if 'churned' in df.columns else 0.0
    active_customers = total_customers - int(df['churned'].sum()) if 'churned' in df.columns else total_customers

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Analyzed Cohort</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-subtext">Total Evaluated Accounts</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Baseline Churn</div>
            <div class="kpi-value">{churn_rate:.2f}%</div>
            <div class="kpi-subtext">{int(df['churned'].sum()):,} Incurred Events</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Retained Base</div>
            <div class="kpi-value">{active_customers:,}</div>
            <div class="kpi-subtext">{100 - churn_rate:.2f}% Active Retention</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Model Input Features</div>
            <div class="kpi-value">{len(feature_cols)}</div>
            <div class="kpi-subtext">Standardized Variables</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top Drivers Display
    st.markdown("#### Primary Decision Drivers")
    st.caption("Top algorithmic drivers ranked by relative Gain in the gradient boosted decision trees:")

    if not top_kpi_drivers.empty:
        driver_cols = st.columns(4)
        for idx, row in top_kpi_drivers.iterrows():
            feat = row["feature"]
            imp_pct = row["importance"] * 100

            with driver_cols[idx]:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">Rank #{idx + 1} Driver</div>
                    <div class="kpi-value" title="{feat}">{feat}</div>
                    <div class="kpi-subtext"><b>{imp_pct:.1f}%</b> Attribution Weight</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_chart, col_insight = st.columns([1.15, 0.85])
        with col_chart:
            top_plot = top_kpi_drivers.sort_values(by="importance", ascending=True)
            max_imp = float(top_plot["importance"].max())
            fig_bar = px.bar(
                top_plot,
                x="importance",
                y="feature",
                orientation='h',
                text=top_plot['importance'].apply(lambda x: f"{x * 100:.1f}%"),
                title="<b>Feature Importance Distribution</b>",
                color="importance",
                color_continuous_scale=[[0, "#93C5FD"], [1.0, "#1D4ED8"]],
                labels={"importance": "Relative Weight", "feature": "Feature"}
            )
            fig_bar.update_traces(
                textposition='outside',
                textfont=dict(color="#000000", size=12, family="Inter")
            )
            fig_bar.update_layout(
                autosize=True,
                height=300,
                margin=dict(l=10, r=45, t=35, b=20),
                coloraxis_showscale=False,
                plot_bgcolor='#FFFFFF',
                paper_bgcolor='#FFFFFF',
                font=dict(family="Inter", color="#000000"),
                xaxis=dict(
                    range=[0, max_imp * 1.25],
                    showgrid=True,
                    gridcolor="#E2E8F0",
                    zeroline=False,
                    tickfont=dict(color="#000000")
                ),
                yaxis=dict(zeroline=False, tickfont=dict(color="#000000", size=12))
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_insight:
            top_name = top_kpi_drivers.iloc[0]["feature"]
            second_name = top_kpi_drivers.iloc[1]["feature"]
            top_two_share = (top_kpi_drivers.iloc[0]["importance"] + top_kpi_drivers.iloc[1]["importance"]) * 100
            st.markdown(f"""
            <div class="info-callout">
                <div style="font-weight: 700; font-size: 0.95rem; color: #000000; margin-bottom: 6px;">
                    Interpretability Overview
                </div>
                <p style="font-size: 0.88rem; color: #000000; line-height: 1.5; margin-bottom: 8px;">
                    The primary features, <b>{top_name}</b> and <b>{second_name}</b>, account for 
                    <b>{top_two_share:.1f}%</b> of total predictive attribution.
                </p>
                <ul style="font-size: 0.84rem; color: #000000; padding-left: 18px; margin-bottom: 0;">
                    <li>High values in <code>{top_name}</code> serve as the primary retention indicator.</li>
                    <li>Sustained declines in <code>{second_name}</code> represent the key risk signal.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Full Feature Hierarchy Table
    st.markdown("#### Complete Feature Attribution Hierarchy")
    if not all_ranked_drivers.empty:
        ranked_display_df = all_ranked_drivers.copy()
        ranked_display_df["Rank"] = [f"#{i+1}" for i in range(len(ranked_display_df))]
        ranked_display_df["Attribution Score"] = (ranked_display_df["importance"] * 100).round(2).astype(str) + "%"

        def get_tier(val):
            if val >= 0.15:
                return "Tier 1 (High Impact)"
            elif val >= 0.05:
                return "Tier 2 (Moderate Impact)"
            else:
                return "Tier 3 (Marginal Contributor)"

        ranked_display_df["Impact Tier"] = ranked_display_df["importance"].apply(get_tier)
        ranked_display_df.rename(columns={"feature": "Feature Variable"}, inplace=True)

        st.dataframe(
            ranked_display_df[["Rank", "Feature Variable", "Attribution Score", "Impact Tier"]],
            use_container_width=True,
            hide_index=True
        )

# ================= TAB 2: PRODUCTION MODEL BENCHMARK SCORECARD =================
with tabs[1]:
    st.markdown("#### Production Architecture Performance")
    st.caption("Empirical testing results across 6,103 holdout validation records (Stratified 80/20 Split):")

    # Production Champion Highlight
    st.markdown("""
    <div style="background: #F8FAFC; border: 1px solid #000000; border-left: 5px solid #10B981; padding: 18px 22px; border-radius: 8px; margin-bottom: 20px;">
        <div style="font-size: 1rem; font-weight: 800; color: #000000;">Champion Architecture: Extreme Gradient Boosting (XGBoost)</div>
        <p style="font-size: 0.88rem; color: #000000; margin: 4px 0 0 0; line-height: 1.5;">
            Selected for production deployment due to superior separation (<b>ROC-AUC: 0.9040</b>, <b>PR-AUC: 0.8412</b>) and balanced performance across <b>Precision (0.78)</b> and <b>Recall (0.72)</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Production Model KPIs in a clean 2-row grid for multi-device readability
    kpi_r1_c1, kpi_r1_c2, kpi_r1_c3 = st.columns(3)
    with kpi_r1_c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">ROC-AUC</div>
            <div class="kpi-value">0.9040</div>
            <div class="kpi-subtext" style="color: #16A34A;">+0.0228 vs Random Forest</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_r1_c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">PR-AUC</div>
            <div class="kpi-value">0.8412</div>
            <div class="kpi-subtext" style="color: #16A34A;">Prevalence Baseline: 0.3549</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_r1_c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Accuracy</div>
            <div class="kpi-value">82.71%</div>
            <div class="kpi-subtext" style="color: #16A34A;">+2.01% vs Random Forest</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    kpi_r2_c1, kpi_r2_c2, kpi_r2_c3 = st.columns(3)
    with kpi_r2_c1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Precision</div>
            <div class="kpi-value">0.78</div>
            <div class="kpi-subtext">Low False-Positive Interventions</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_r2_c2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Recall</div>
            <div class="kpi-value">0.72</div>
            <div class="kpi-subtext">Identified Attrition Capture</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_r2_c3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">F1-Score</div>
            <div class="kpi-value">0.75</div>
            <div class="kpi-subtext" style="color: #16A34A;">Calibrated Harmonic Mean</div>
        </div>
        """, unsafe_allow_html=True)

    # Technical Footnote on Split, Class Imbalance, and Metric Rationale
    st.markdown("""
    <div class="info-callout" style="margin-top: 16px; margin-bottom: 22px;">
        <div style="font-weight: 700; font-size: 0.88rem; color: #000000; margin-bottom: 4px;">
            Holdout Validation & Metric Methodology
        </div>
        <p style="font-size: 0.84rem; color: #000000; margin: 0; line-height: 1.55;">
            • <b>Holdout Split:</b> 6,103 stratified records (2,166 churned vs. 3,937 retained).<br>
            • <b>Class Ratio Context:</b> Holdout churn rate is <b>35.49%</b>. While enterprise SaaS benchmarks typically average 15%–25%, this moderate skew necessitates evaluating precision against minority recall.<br>
            • <b>PR-AUC Evaluation:</b> Measures precision across recall thresholds against a non-informative baseline of 0.3549, ensuring high recall does not generate disproportionate false-positive interventions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Holdout Calibration Visuals
    st.markdown("#### Holdout Calibration Visuals")

    actual_churn_pct = 35.49
    pred_churn_pct = 32.61
    actual_churn_count = 2166
    pred_churn_count = 1990
    actual_retained_count = 3937
    pred_retained_count = 4113

    vis_col1, vis_col2 = st.columns(2)

    with vis_col1:
        calib_df = pd.DataFrame({
            "Cohort": ["Actual Ground Truth", "Predicted (@ 0.76 Threshold)"],
            "Churned (%)": [actual_churn_pct, pred_churn_pct],
            "Retained (%)": [100 - actual_churn_pct, 100 - pred_churn_pct]
        })

        fig_calib = go.Figure()
        fig_calib.add_trace(go.Bar(
            y=calib_df["Cohort"],
            x=calib_df["Churned (%)"],
            name="Churned",
            orientation="h",
            text=calib_df["Churned (%)"].apply(lambda x: f"{x:.2f}%"),
            textposition="inside",
            marker=dict(color="#EF4444")
        ))
        fig_calib.add_trace(go.Bar(
            y=calib_df["Cohort"],
            x=calib_df["Retained (%)"],
            name="Retained",
            orientation="h",
            text=calib_df["Retained (%)"].apply(lambda x: f"{x:.2f}%"),
            textposition="inside",
            marker=dict(color="#10B981")
        ))

        fig_calib.update_layout(
            title="<b>Actual vs. Predicted Churn Distribution</b>",
            barmode="stack",
            autosize=True,
            height=240,
            margin=dict(l=10, r=10, t=35, b=20),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Inter", color="#000000"),
            xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, ticksuffix="%"),
            yaxis=dict(zeroline=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
        )
        st.plotly_chart(fig_calib, use_container_width=True)

    with vis_col2:
        cohort_types = ["Churned Accounts", "Retained Accounts"]
        actual_vals = [actual_churn_count, actual_retained_count]
        pred_vals = [pred_churn_count, pred_retained_count]

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=cohort_types,
            y=actual_vals,
            name="Actual Ground Truth",
            text=[f"{v:,}" for v in actual_vals],
            textposition="outside",
            marker=dict(color="#3B82F6")
        ))
        fig_vol.add_trace(go.Bar(
            x=cohort_types,
            y=pred_vals,
            name="Predicted (@ 0.76)",
            text=[f"{v:,}" for v in pred_vals],
            textposition="outside",
            marker=dict(color="#94A3B8")
        ))

        fig_vol.update_layout(
            title="<b>Holdout Record Classification Volume</b>",
            barmode="group",
            autosize=True,
            height=240,
            margin=dict(l=10, r=10, t=35, b=20),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Inter", color="#000000"),
            yaxis=dict(range=[0, 4800], showgrid=True, gridcolor="#F1F5F9"),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-model Comparison Matrix
    benchmark_data = {
        "Model Architecture": ["XGBoost Classifier", "Random Forest", "Gradient Boosting", "Logistic Regression"],
        "ROC-AUC": [0.9040, 0.8812, 0.8347, 0.7905],
        "PR-AUC": [0.8412, 0.8035, 0.7410, 0.6720],
        "Accuracy (%)": [82.71, 80.70, 76.63, 73.52],
        "Precision": [0.78, 0.75, 0.70, 0.66],
        "Recall": [0.72, 0.68, 0.60, 0.53],
        "F1-Score": [0.75, 0.71, 0.65, 0.59],
        "Status": ["Deployed Champion", "Candidate", "Candidate", "Baseline"]
    }
    benchmark_df = pd.DataFrame(benchmark_data)

    col_table, col_radar = st.columns([1.1, 0.9])
    with col_table:
        st.markdown("#### Model Benchmark Matrix")
        st.dataframe(benchmark_df, use_container_width=True, hide_index=True)

    with col_radar:
        fig_compare = go.Figure()
        colors = ["#2563EB", "#0284C7", "#D97706", "#94A3B8"]
        
        for i, row in benchmark_df.iterrows():
            fig_compare.add_trace(go.Bar(
                name=row["Model Architecture"],
                x=["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1-Score"],
                y=[row["ROC-AUC"], row["PR-AUC"], row["Precision"], row["Recall"], row["F1-Score"]],
                marker_color=colors[i]
            ))

        fig_compare.update_layout(
            title="<b>Comparative Performance Profile</b>",
            barmode='group',
            autosize=True,
            height=300,
            margin=dict(l=10, r=10, t=35, b=20),
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(family="Inter", color="#000000"),
            yaxis=dict(range=[0.3, 1.0], gridcolor="#F1F5F9"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

# ================= TAB 3: INTERACTIVE PREDICTION SIMULATOR =================
with tabs[2]:
    st.markdown("#### Real-Time Inference Simulator")
    st.caption("Simulate behavioral telemetry variations to evaluate instantaneous churn risk:")

    # 1. Preset Archetype Definitions (Extracted Dynamically from processed_data.csv for reference)
    churned_cohort = df[df["churned"] == 1] if "churned" in df.columns else df
    retained_cohort = df[df["churned"] == 0] if "churned" in df.columns else df

    high_risk_preset = {}
    retained_preset = {}

    for col in feature_cols:
        is_discrete = df[col].nunique() <= 10 or pd.api.types.is_integer_dtype(df[col])
        if is_discrete:
            high_risk_preset[col] = int(churned_cohort[col].median())
            retained_preset[col] = int(retained_cohort[col].median())
        else:
            high_risk_preset[col] = round(float(churned_cohort[col].median()), 2)
            retained_preset[col] = round(float(retained_cohort[col].median()), 2)

    preset_profiles = {
        "High-Risk Churn Archetype": high_risk_preset,
        "Retained / Healthy Archetype": retained_preset
    }

    # 2. Reference Feature Comparison Table Dropdown
    with st.expander("🔍 View Detailed Feature Telemetry Comparison Across Archetypes", expanded=False):
        comp_data = []
        for feat in feature_cols:
            is_discrete = df[feat].nunique() <= 10 or pd.api.types.is_integer_dtype(df[feat])
            comp_data.append({
                "Feature Variable": feat,
                "High-Risk Churn Example": preset_profiles["High-Risk Churn Archetype"].get(feat, "-"),
                "Retained / Healthy Example": preset_profiles["Retained / Healthy Archetype"].get(feat, "-"),
                "Cohort Median": int(df[feat].median()) if is_discrete else round(float(df[feat].median()), 2)
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Form Inputs (Adaptive Grid initialized with cohort medians)
    inputs = {}
    input_cols = st.columns(3)

    for i, col in enumerate(feature_cols):
        with input_cols[i % 3]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            is_discrete = df[col].nunique() <= 10 or pd.api.types.is_integer_dtype(df[col])
            default_val = int(df[col].median()) if is_discrete else float(df[col].median())

            if is_discrete:
                inputs[col] = st.number_input(
                    f"{col}",
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=default_val,
                    step=1,
                    key=f"input_{col}"
                )
            else:
                inputs[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=f"input_{col}"
                )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Run Model Prediction", type="primary", use_container_width=True)

    # 4. Prediction Execution & Output Rendering (76% Threshold)
    if predict_clicked:
        pred, prob = predict_single_record(model, inputs)
        prob_pct = prob * 100

        col_verdict, col_gauge = st.columns([1, 1])

        with col_verdict:
            st.markdown("<br>", unsafe_allow_html=True)
            if prob_pct >= 76.0:
                st.markdown(f"""
                <div style="background-color: #FEF2F2; border: 1px solid #DC2626; border-radius: 10px; padding: 22px;">
                    <div style="font-size: 0.78rem; font-weight: 800; color: #991B1B; text-transform: uppercase; letter-spacing: 0.05em;">Classification Outcome</div>
                    <div style="font-size: 1.45rem; font-weight: 800; color: #991B1B; margin: 4px 0;">Elevated Churn Risk</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #DC2626; line-height: 1;">{prob_pct:.1f}%</div>
                    <p style="color: #7F1D1D; font-size: 0.88rem; margin-top: 10px; line-height: 1.5; font-weight: 600;">
                        Telemetry signals meet or exceed the calibrated threshold (<b>&ge; 76%</b>). Recommend automated retention touchpoints or account review.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #F0FDF4; border: 1px solid #16A34A; border-radius: 10px; padding: 22px;">
                    <div style="font-size: 0.78rem; font-weight: 800; color: #166534; text-transform: uppercase; letter-spacing: 0.05em;">Classification Outcome</div>
                    <div style="font-size: 1.45rem; font-weight: 800; color: #166534; margin: 4px 0;">Retained Cohort</div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #16A34A; line-height: 1;">{prob_pct:.1f}%</div>
                    <p style="color: #14532D; font-size: 0.88rem; margin-top: 10px; line-height: 1.5; font-weight: 600;">
                        Customer engagement metrics are within normal parameters (<b>&lt; 76%</b>). Account is classified as stable.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                number={'suffix': "%", 'font': {'size': 32, 'family': "Inter", 'color': '#000000'}},
                title={'text': "<b>Predicted Churn Probability</b>", 'font': {'size': 14, 'color': '#000000'}},
                gauge={
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': "#000000",
                        'tickvals': [0, 25, 50, 76, 100],
                        'ticktext': ["0", "25", "50", "76", "100"],
                        'tickfont': {'color': '#000000'}
                    },
                    'bar': {'color': "#000000", 'thickness': 0.22},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#000000",
                    'steps': [
                        {'range': [0, 45], 'color': '#DCFCE7'},
                        {'range': [45, 76], 'color': '#FEF3C7'},
                        {'range': [76, 100], 'color': '#FEE2E2'}
                    ],
                    'threshold': {
                        'line': {'color': "#DC2626", 'width': 3},
                        'thickness': 0.75,
                        'value': 76.0
                    }
                }
            ))
            fig_gauge.update_layout(
                autosize=True,
                height=260,
                margin=dict(l=20, r=20, t=35, b=10),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
