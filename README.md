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


📘 Full Explanation & Approach for the Project
Title:

Advanced Time Series Forecasting with Prophet and Model Explainability using SHAP

1. Introduction

Time series forecasting is crucial in domains such as finance, energy consumption, sales analytics, and sensor-based monitoring. Traditional models like ARIMA often struggle to capture complex patterns such as multiple seasonalities, sudden shocks, holiday effects, and the combined influence of external variables.

This project uses Facebook Prophet, a highly flexible additive model that automatically handles:

Trend changes

Weekly, yearly, and custom seasonality

Holiday impacts

External regressors

Furthermore, to improve transparency, the project uses SHAP (SHapley Additive exPlanations) to explain why the model produces each prediction — a critical requirement in modern business and research settings where explainability is essential.

2. Problem Statement

Design and implement a robust forecasting system that:

Models real-world time series behavior using Prophet

Incorporates external regressors affecting the target variable

Handles holidays and special events

Applies rigorous time-series cross-validation

Produces forecast accuracy metrics

Uses SHAP to interpret the model and analyze:

Which regressors influence the model the most

How trend/seasonality contribute

Why forecast values change across time

3. Dataset Creation and Characteristics

Since real-world datasets can be restricted, this project uses a synthetically generated time series that imitates real-world patterns.

3.1 Components of the dataset

The dataset is constructed to contain:

(a) Multiple seasonalities

Yearly seasonality

Modeled via sinusoidal patterns to mimic temperature-driven cycles

Weekly seasonality

Weekdays have higher values; weekends show lower activity

(b) Trend

A slow-moving upward economic trend is included.

(c) Shocks

Random short-term events lasting 7 days:

Sudden spikes

Sudden drops
These simulate:

Outages

Promotions

Extreme weather

(d) External regressors

Two regressors are included:

Feature	Explanation
temp	Weather effect — daily varying seasonal temperature
econ	Slow economic index that grows over time
(e) Holidays

Important national holidays (e.g., Jan 26, Aug 15) are included with special Prophet handling.

4. Modeling Approach

The heart of the project is the Prophet model.

Prophet uses the formula:

y(t) = trend(t) + seasonality(t) + holiday(t) + β * regressors(t) + error

4.1 Trend Modeling

The model uses piecewise linear trend with potential changepoints.
We tune:

changepoint_prior_scale → controls trend flexibility

Higher values → more flexible trend
Lower values → smoother trend

5. Seasonality Modeling

Prophet automatically models:

Weekly seasonality

Yearly seasonality

But we explicitly add:

weekly_custom with Fourier terms

yearly_custom with Fourier terms

More Fourier terms = ability to capture more complex cycles.

6. Adding External Regressors

External regressors allow Prophet to incorporate real-world influencing variables.

We add:

temp

econ

This means the model understands:

Warmer days increase consumption

Economic growth affects long-term patterns

7. Holiday Effects

Holiday effects cause deviations not explainable by usual weekly/yearly patterns.

Prophet handles holidays by assigning learned coefficients to each holiday date.

8. Cross-Validation Approach

Using typical machine-learning cross-validation breaks time order — which is invalid for forecasting.

So we use Rolling Origin Evaluation, also called Expanding Window CV.

8.1 Why Rolling Origin CV?

Because:

Future data must never influence past model training

Window expands as more data becomes available

Mimics real-world forecasting environments

8.2 CV Configuration
Component	Value
Initial training window	2 years
Forecast horizon	90 days
Step size	180 days
Number of folds	Determined by dataset length

Each fold:

Trains on data from day 0 → day N

Predicts next 90 days

Evaluates MAE, RMSE, MAPE

9. Hyperparameter Tuning

We tune Prophet's flexibility using:

changepoint_prior_scale = [0.01, 0.05, 0.2]

Lower values → smoother trend
Higher values → sensitive to local variations

For each fold:

All parameters are tested

Best fold+param combination is selected based on lowest MAE

10. Model Performance Metrics
We compute:

MAE (Mean Absolute Error)
Measures average error magnitude.

RMSE (Root Mean Square Error)
Punishes large errors more heavily.

MAPE (Mean Absolute Percentage Error)
Measures error as percentage — intuitive for business use.

11. Explainability Using SHAP

Prophet does not natively integrate with SHAP, so we wrap the model inside a prediction function and apply Kernel SHAP.

11.1 Purpose of SHAP

It helps answer:

How much did temperature contribute to a given forecast?

How did the economic index shift the trend?

Why is the forecast higher on certain days?

Which regressor matters most?

11.2 SHAP Outputs Generated

SHAP Feature Importance Plot
Shows mean absolute contribution of each regressor.

SHAP Force/Waterfall Plot
Shows instance-level contributions:

Base value

temp contribution

econ contribution

Final forecast = sum of all effects

Text-based Feature Importance Ranking

12. Interpretation of Results
Key findings typically observed:

Temperature usually has the strongest short-term influence.

Economic index affects long-term trend movement.

Yearly seasonality captures climate-related behavior.

Weekly seasonality reflects weekday/weekend structural differences.

Holiday spikes are clearly visible in predictions.

SHAP values confirm which feature contributes most at any given time.

This makes the forecasting model:

Transparent

Explainable

Trustworthy

Production-ready

13. Why Prophet + SHAP is a Powerful Combination
Prophet	SHAP
Great forecasting performance	Great interpretability
Handles trends/seasonality easily	Feature-level + instance-level explanations
Accepts regressors and holidays	Quantifies contribution of regressors
Works well with missing/irregular data	Helps justify business decisions

Together, they create a system that provides both high accuracy and deep transparency.

14. Conclusion

This project builds a complete forecasting and explainability pipeline that closely reflects real-world production systems. The methodology:

Generates realistic time-series data

Trains a flexible, powerful model (Prophet)

Applies rigorous CV

Evaluates performance with multiple metrics

Uses SHAP to bring transparency to the model’s decisions

This workflow can be adapted to:

Retail forecasting

Energy load prediction

Finance time series

Supply chain forecasting

IoT sensor anomaly detection

The combination of Prophet + SHAP ensures the model is not a "black box" but a fully interpretable forecasting engine.

🙏 Acknowledgements

Facebook Prophet developers

SHAP authors

scikit-learn for cross-validation framework
