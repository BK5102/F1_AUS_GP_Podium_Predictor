# F1 AUS GP Podium Predictor

A machine learning model that predicts the probability of a podium finish 
at the 2026 Australian Grand Prix, built in one week.

## Project Goal
Predict which drivers will finish in the top 3 at the 2026 AUS GP season 
opener using 2023–2025 historical data. Predictions will be published 
before the race and compared to actual results after.

## Deliverables
- Podium probability estimates per driver
- SHAP explainability plot
- Streamlit dashboard
- Blog post explaining modeling decisions

## Data Sources
- FastF1 (race results, qualifying positions, lap data)

## Tech Stack
Python, FastF1, XGBoost, SHAP, Streamlit, scikit-learn

## Repo Structure
F1_AUS_GP_Podium_Predictor/
├── 01_data_collection.ipynb  # Main notebook
├── aus_gp_data.csv         # Cleaned race data
├── requirements.txt        # Dependencies
└── README.md

## How to Run
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Open podium_predictor.ipynb in JupyterLab
```
