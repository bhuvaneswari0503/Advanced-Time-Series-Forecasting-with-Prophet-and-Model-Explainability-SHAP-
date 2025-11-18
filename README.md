# Advanced-Time-Series-Forecasting-with-Prophet-and-Model-Explainability-SHAP-

Advanced Time Series Forecasting with Prophet and Model Explainability (SHAP)
📌 Project Overview

This project implements an advanced end-to-end time series forecasting workflow using Facebook Prophet, enhanced with:

Multiple seasonalities

External regressors (weather, economic index, etc.)

Holiday effects

Rolling-origin time-series cross-validation

Model performance evaluation (MAE, RMSE, MAPE)

SHAP-based model explainability to quantify how trend, seasonality, and regressors influence predictions

The goal is to produce a fully interpretable forecasting model, demonstrating both predictive performance and transparent explanation of forecasts.

📂 Project Structure
project/
│
├── data/
│   └── synthetic_timeseries_prophet.csv
│
├── notebooks/
│   └── prophet_shap_analysis.ipynb      # Full notebook version
│
├── src/
│   └── prophet_shap_pipeline.py         # Full Python script
│
├── outputs/
│   ├── prophet_model/
│   │   └── model.json                   # Serialized Prophet model
│   ├── shap_feature_importance.png
│   ├── shap_force_approx.png
│   └── prophet_shap_report.txt
│
└── README.md

📊 Dataset Description

The dataset (synthetic) includes:

Column	Description
ds	Timestamp (daily)
y	Target time series (energy-like demand)
temp	External weather regressor
econ	Slow-moving economic trend regressor

The dataset also embeds weekly, yearly, and shock patterns + holiday effects.

🧠 Modeling Approach
Prophet Model Components

The model includes:

Additive trend

Weekly seasonality

Yearly seasonality

National holiday effects

External regressors:

temp

econ

Cross-Validation Strategy

A rolling-origin expanding-window CV is used:

Initial training window: 2 years

Forecast horizon: 90 days

Step size: 180 days

Each fold trains a new Prophet model and computes:

MAE

RMSE

MAPE

The model with the lowest MAE is selected as the final model.

📈 Metrics Reported

For the final selected fold:

Mean Absolute Error (MAE)

Root Mean Square Error (RMSE)

Mean Absolute Percentage Error (MAPE)

🔍 SHAP Explainability

To understand why the model predicts what it predicts:

Kernel SHAP is applied to the final Prophet model

SHAP explains the contribution of:

External regressors (temp, econ)

Trend

Seasonalities

Outputs include:

SHAP feature importance plot

SHAP waterfall/force plot for individual predictions

Text summary comparing relative contributions

⚙️ Requirements

Install dependencies:

pip install pandas numpy matplotlib scikit-learn prophet shap


Some environments require installing cmdstanpy before Prophet.
If you face issues:
pip install cmdstanpy

▶️ How to Run the Project
Option 1 — Use the Python script
python src/prophet_shap_pipeline.py


This will:

Generate dataset

Train models + perform CV

Select best model

Run SHAP explainability

Save all outputs to outputs/

Option 2 — Use the Jupyter Notebook

Open:

notebooks/prophet_shap_analysis.ipynb


Then run each cell step-by-step.

📄 Outputs Generated

The pipeline produces the following artifacts:

File	Description
prophet_model/model.json	Serialized Prophet model
shap_feature_importance.png	SHAP summary bar plot
shap_force_approx.png	Waterfall-style SHAP force plot
prophet_shap_report.txt	Full textual report
synthetic_timeseries_prophet.csv	Full dataset
📝 Summary of Findings (Example)

Temperature contributed the most variability to short-term forecasts

Economic index influenced the long-term trend

Yearly and weekly seasonality added predictable recurring structure

Holidays had small but noticeable effects on specific dates

The SHAP analysis clearly decomposed the forecast into understandable parts

🙏 Acknowledgements

Facebook Prophet developers

SHAP authors

scikit-learn for cross-validation framework
