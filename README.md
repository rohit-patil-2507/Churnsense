# Churnsense
# 🎯 ChurnSense: Enterprise Retention & Risk Intelligence Hub  > An end-to-end customer churn prediction and retention analytics platform that translates behavioral telemetry into early attrition alerts, powered by an optimized XGBoost classifier and an executive decision dashboard.  ---  ## 📌 Executive Summary  Customer churn represents a critical revenue leak for subscription-based business models. **ChurnSense** delivers a production-grade churn intelligence system designed to detect early churn signals before cancellation occurs.  Rather than relying on uncalibrated baseline predictions, ChurnSense utilizes an imbalanced-aware machine learning pipeline evaluated against **PR-AUC (0.8412)** and **ROC-AUC (0.9040)** on holdout validation data, coupled with a calibrated **76% decision threshold** to minimize expensive false-positive retention outreach.  ---  ## 🏗️ System Architecture  ```text ├── data/ │   ├── raw/cleaned_data.csv          # Raw merged subscription & usage records (30,515 rows) │   └── processed_data.csv            # Cleaned, model-ready feature engineered dataset ├── notebooks/ │   ├── 01_eda.ipynb                  # Exploratory data analysis & feature distributions │   ├── 02_feature_engineering.ipynb  # Behavioral aggregation & categorical encoding │   └── 03_model_building.ipynb       # Stratified cross-validation, tuning & export ├── models/ │   └── best_model.pkl                # Production XGBoost artifact & threshold metadata ├── utils/ │   ├── data_utils.py                 # Dynamic dataset loader & feature schema extractors │   └── model_utils.py                # Single-record inference & feature attribution engine ├── app.py                            # Multi-tab responsive Streamlit executive interface └── requirements.txt                  # Production dependencies  ```  ---  ## ⚙️ Technical Stack & Frameworks  * **Backend & Data Processing**: Python, NumPy, Pandas, Scikit-learn * **Machine Learning Architecture**: XGBoost Classifier (Extreme Gradient Boosting) * **Validation & Metrics**: StratifiedKFold Cross-Validation, ROC-AUC, PR-AUC (Average Precision), F1-Score * **Frontend & Visualization**: Streamlit, Plotly Express, Plotly Graph Objects * **Design System**: Responsive mobile/tablet/desktop CSS grid with high-contrast accessibility  ---  ## 📊 Model Performance & Holdout Benchmark  Evaluated on an **80/20 stratified holdout split** consisting of **6,103 records** (2,166 churned vs. 3,937 retained, 35.49% baseline churn rate):  | Model Architecture | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1-Score | Operational Status | | --- | --- | --- | --- | --- | --- | --- | --- | | **XGBoost Classifier** | **0.9040** | **0.8412** | **82.71%** | **0.78** | **0.72** | **0.75** | 🏆 **Deployed Champion** | | Random Forest | 0.8812 | 0.8035 | 80.70% | 0.75 | 0.68 | 0.71 | Candidate Benchmark | | Gradient Boosting | 0.8347 | 0.7410 | 76.63% | 0.70 | 0.60 | 0.65 | Candidate Benchmark | | Logistic Regression | 0.7905 | 0.6720 | 73.52% | 0.66 | 0.53 | 0.59 | Linear Baseline |  ### 🔍 Precision-Recall AUC (PR-AUC) Rationale  In subscription retention, a standard 50% cutoff often triggers high false-positive outreach costs. Evaluating **PR-AUC (0.8412)** against the baseline class prevalence of **0.3549** ensures that customer success interventions capture true churn risk with high confidence, supported by the production threshold set at **≥ 76%**.  ---  ## 🖥️ Executive Dashboard Modules (`app.py`)  1. **Executive Dashboard**: * High-level portfolio tracking: 30,515 monitored accounts, 35.49% baseline churn rate, and 12 model features. * Algorithmic feature attribution isolating core decision drivers (`subscription_year`, `signup_year`, `plan_type`, and `signup_month`). * Interactive horizontal importance ranking and complete attribution hierarchy tables.   2. **Model Performance & Validation**: * Comprehensive model benchmark comparison matrix across 4 algorithms. * Proportional actual vs. predicted churn rate distributions. * Volume parity charts validating holdout classification counts.   3. **Single-Customer Simulator**: * Real-time parameter simulation across behavioral telemetry features. * Collapsible telemetry comparison breakdown contrasting high-risk vs. retained archetypes. * Dynamic classification gauge indicator calibrated to the **76% critical threshold**.    ---  ## 🚀 Quickstart & Local Setup  ```bash # 1. Clone repository git clone https://github.com/rohit-patil-2507/ChurnSense.git cd churnsense  # 2. Create and activate virtual environment python -m venv .venv source .venv/bin/activate  # On Windows: .venv\Scripts\activate  # 3. Install dependencies pip install -r requirements.txt  # 4. Launch the dashboard streamlit run app.py  ```
# 🎯 ChurnSense: Enterprise Retention & Risk Intelligence Hub

> An end-to-end customer churn prediction and retention analytics platform that translates behavioral telemetry into proactive attrition warnings, powered by an optimized XGBoost classifier and an executive Streamlit decision dashboard.

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Scikit--Learn-orange.svg?logo=scikit-learn&logoColor=white)](https://xgboost.readthedocs.io/)
[![Frontend](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model Performance](https://img.shields.io/badge/ROC--AUC-0.9040-success.svg)](https://scikit-learn.org/)
[![Precision-Recall](https://img.shields.io/badge/PR--AUC-0.8412-brightgreen.svg)](https://scikit-learn.org/)

---

## 📌 Executive Summary & Problem Formulation

Customer churn directly erodes recurring subscription revenues and drives up Customer Acquisition Cost (CAC) burdens. **ChurnSense** is engineered as a modular, production-grade intelligence layer that analyzes behavioral usage patterns and subscription telemetry to forecast customer attrition before cancellation decisions materialize.

### Core Challenges Addressed:
* **Class Imbalance & Uncalibrated Defaults**: Standard classifiers evaluated on accuracy alone fall prey to majority-class bias. A standard 0.50 cutoff triggers excessive false alarms, inflating Customer Success (CS) intervention overhead.
* **Complex Telemetry Patterns**: Identifying the compound effects of reduced in-app usage (`minutes_active`), declining engagement (`classes_booked`), escalating support interactions (`support_tickets`), and cohort aging.
* **Executive Decision Bridging**: Converting raw machine learning probabilities into interpretable, ranked feature attributions and risk dials tailored for portfolio leadership and CS operations.

---

## 📂 Data Source & Origin

* **Primary Dataset Source**: [Data Career School — Customer Churn Analysis DataLab](https://www.datacareerschool.com/datalabs/customer-churn-analysis)
* **Dataset Scope**: Comprehensive recurring subscription and user behavior telemetry, tracking account lifecycles, service tiers, app engagement, and cancellation records across 30,515 customer profiles.

---

## 🏗️ System Architecture & File Structure

```text
ChurnSense/
├── data/
│   ├── raw/
│   │   ├── subscriptions.csv        # Raw account subscription histories
│   │   ├── usage.csv                # In-app telemetry & activity metrics
│   │   ├── merged.csv               # Merged customer profile records
│   │   └── cleaned_data.csv         # Processed raw dataset for modeling
│   └── processed_data.csv           # Model-ready dataset (30,515 records, 12 features)
├── models/
│   └── best_model.pkl               # Serialized champion XGBoost model & inference metadata
├── notebooks/
│   ├── 01_eda.ipynb                 # Telemetry exploration, missing value imputation & encoding
│   └── 03_model_building.ipynb      # Stratified 5-fold CV, tuning & threshold optimization
├── utils/
│   ├── __init__.py                  # Utility package initializer
│   ├── data_utils.py                # Processed dataset loader & feature schema extractors
│   └── model_utils.py               # Real-time inference & feature attribution engine
├── app.py                           # Multi-tab responsive Streamlit executive interface
├── requirements.txt                 # Project environment dependencies
├── template.py                      # Automated project scaffolding script
└── README.md                        # Production documentation & project manual
```

---

## ⚙️ Technical Stack & Tooling

* **Data Engineering & Analysis**: Python, NumPy, Pandas
* **Machine Learning & Tuning**: XGBoost, Scikit-learn (`StratifiedKFold`, `RandomizedSearchCV`)
* **Evaluation Framework**: ROC-AUC, PR-AUC (`average_precision_score`), Precision, Recall, F1-Score
* **Interactive UI & Visualizations**: Streamlit, Plotly Express, Plotly Graph Objects
* **Deployment & Formatting**: Custom CSS Grid Layouts, Pickle Serialization, Git Version Control

---

## 📊 End-to-End Pipeline & Engineering Methodology

### 1. Data Processing & Behavioral Feature Extraction (`01_eda.ipynb`)
* **Multi-Source Telemetry Merging**: Relational inner join between subscription account metadata and platform activity logs (`customer_id`).
* **Categorical Encoding Hierarchy**:
  * `plan_type`, `acquisition_channel`, `primary_device`: Transformed via Label Encoding with deterministic integer mappings.
  * `age_group`, `monthly_price`: Transformed via Ordinal Encoding to preserve progressive bucket rankings.
* **Temporal Decomposition**:
  * Deconstructed string timestamps (`month`, `signup_date`) into cyclical and sequential numerical dimensions (`subscription_year`, `subscription_month`, `signup_year`, `signup_month`).
  * Enables the model to isolate seasonal cancellations and tenure-based attrition rates.
* **Multicollinearity Elimination**:
  * Pearson correlation analysis highlighted high collinearity between `plan_tier` and `monthly_price`. Redundant categorical variables were removed to preserve tree variance and split stability.
  * Dropped post-churn artifacts (`churn_date`, `cancel_reason`) to eliminate data leakage.

### 2. Feature Schema & Data Dictionary

The final production dataset (`data/processed_data.csv`) contains **30,515 records** and **12 input features**:

| Feature Variable | Data Type | Encoding Strategy | Behavioral Description |
| :--- | :---: | :---: | :--- |
| `subscription_year` | Integer | Temporal Year | Year of active subscription period under observation |
| `subscription_month` | Integer | Cyclical Month | Month of subscription renewal/billing cycle (1–12) |
| `signup_year` | Integer | Temporal Year | Account origination year |
| `signup_month` | Integer | Cyclical Month | Account origination calendar month (1–12) |
| `plan_type` | Integer | Label Encoded | Subscription service tier |
| `monthly_price` | Float | Ordinal Binned | Normalized pricing bracket of the active subscription |
| `acquisition_channel` | Integer | Label Encoded | Attribution channel (Organic, Paid Search, Referral, Social) |
| `primary_device` | Integer | Label Encoded | Primary hardware access point (iOS, Android, Web) |
| `age_group` | Float | Ordinal Binned | User demographic cohort tier |
| `minutes_active` | Float | Continuous | Cumulative monthly platform interaction time (minutes) |
| `classes_booked` | Integer | Discrete Count | Total sessions/classes reserved within billing window |
| `support_tickets` | Integer | Discrete Count | Incurred customer support incident logs |
| **`churned`** | **Binary** | **Target Class** | **Ground truth attrition indicator (0 = Retained, 1 = Churned)** |

---

## 🏆 Model Benchmarking & Experimental Results

Models were trained and evaluated on an **80/20 Stratified Train/Test Split** to maintain proportional class distributions (35.49% churn rate):
* **Training Cohort**: 24,412 records
* **Holdout Validation Cohort**: 6,103 records (2,166 churned vs. 3,937 retained)

### Multi-Model Comparison Scorecard:

| Architecture | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Accuracy (%) | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** | **0.8871** | **0.9040** | **0.8412** | **82.71%** | **0.78** | **0.72** | **0.75** | 🏆 **Champion** |
| Random Forest | 0.8725 | 0.8812 | 0.8035 | 80.70% | 0.75 | 0.68 | 0.71 | Candidate |
| Gradient Boosting | 0.8210 | 0.8347 | 0.7410 | 76.63% | 0.70 | 0.60 | 0.65 | Candidate |
| Logistic Regression | 0.7780 | 0.7905 | 0.6720 | 73.52% | 0.66 | 0.53 | 0.59 | Linear Baseline |


### Metrics & Threshold Calibration:
* **PR-AUC (0.8412 vs. 0.3549 Baseline)**: On moderately imbalanced datasets, ROC-AUC can present an overly optimistic score. Evaluating Precision-Recall AUC against the naive prevalence line confirms high positive-predictive accuracy across recall thresholds.
* **Calibrated Decision Boundary (Threshold = 0.76)**: In high-touch SaaS/retention settings, offering retention incentives (discounts, dedicated CSM outreach) is expensive. Deploying an operational threshold of **0.76** protects against false positives, reserving outbound workflows for accounts displaying critical attrition risk.

---

## 🖥️ Executive Interface Modules (`app.py`)

The Streamlit dashboard provides three distinct functional layers:

### Module 1: Executive Dashboard
* **Portfolio Level KPIs**: Real-time aggregation of active accounts (30,515), portfolio baseline churn rate (35.49%), and active retention base (19,686 accounts).
* **Algorithmic Attribution Ranking**: Displays tree gain importance scores isolating primary decision drivers:
  * `subscription_year` (18.9% Attribution Weight)
  * `signup_year` (15.6% Attribution Weight)
  * `plan_type` (9.6% Attribution Weight)
  * `signup_month` (7.8% Attribution Weight)
* **Tiered Feature Hierarchy**: Classifies all 12 input features into Tier 1 (High Impact), Tier 2 (Moderate Impact), and Tier 3 (Marginal Contributor) buckets.

### Module 2: Model Performance & Validation
* **Benchmark Scorecard**: Direct tabular comparison across all baseline and ensemble models.
* **Holdout Parity Distribution**: Visual stacked bars validating actual vs. predicted churn proportions.
* **Record Volume Comparison**: Grouped classification counts comparing ground-truth outcomes to model predictions.

### Module 3: Single-Customer Simulator
* **Interactive Telemetry Matrix**: Dynamic input fields allowing account managers to run hypothetical what-if scenarios.
* **Archetype Benchmark Card**: Built-in reference showing median parameter values for typical churned vs. retained accounts.
* **Real-Time Calibrated Gauge**: Interactive radial dial displaying individual risk percentages against the 76% critical intervention boundary.

---

## 🚀 Quickstart & Setup Guide

### Prerequisites
* Python 3.10 or higher
* Git

### Step-by-Step Installation

```bash
# 1. Clone this repository
git clone [https://github.com/rohit-patil-2507/ChurnSense.git](https://github.com/rohit-patil-2507/ChurnSense.git)
cd ChurnSense

# 2. Configure a virtual environment
# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows:
python -m venv .venv
.venv\Scripts\activate

# 3. Upgrade pip and install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the local Streamlit application
streamlit run app.py
```

After starting, access the local dashboard at: `http://localhost:8501`

---

## 👤 Author -[Rohit Patil]
* **LinkedIn**: [Rohit Patil](https://www.linkedin.com/in/rohit-patil-646074361)
* **Email**: [work.patilrohit@gmail.com](mailto:work.patilrohit@gmail.com)
* **Domain Focus**: Data Analytics & Machine Learning Engineering
* **Role**: Data Scientist