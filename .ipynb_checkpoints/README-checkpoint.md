# F1 AUS GP Podium Predictor

A machine learning model that predicts the probability of a podium finish at the 2026 Australian Grand Prix, built in one week.

## Goal
Predict which drivers will finish in the top 3 at the 2026 AUS GP season opener using 2023–2025 historical data. Predictions will be published before the race and compared to actual results after.

## Key Results
* Model predicted that Ferrari has the highet podium probability. Both drivers have AUS podium history AND Ferrari has strong constructor history.
* Grid Position is the most important predictor of the podium finish. Historical record matters less and strong constructor history simply adds          marginal value. 
* If a driver is not starting in the top 5 with good history, the model shows that the podium chances are very low too.
* The SHAP values shows the impact of each feature on the model. The probabilty shows how strongly each feature contributed. The Grid Position feature   had a negative value, meaning it's reducing the "not podium" probability. The Driver Podium Rate feature had a positive value, meaning this record     of AUS history increased the probability.
  
### Explainability Interpretation of the SHAP plot
* The model uses three features to generate each prediction: grid position, the driver's historical podium rate at the Australian GP, and the constructor's historical podium rate.
* Grid position is the strongest predictor. Blue dots representing front-row starters spread far to the right of the chart, meaning starting near the front pushes podium probability up significantly. Pink dots representing back-of-grid starters cluster to the left, pushing probability down.
* The driver's historical podium rate adds a secondary boost. Pink dots representing drivers with strong track records here sit to the right, confirming that past success at this circuit increases predicted probability.
* Constructor history contributes the least, with almost all dots clustered near zero regardless of color, meaning team performance at this circuit adds minimal predictive value.
* Overall, the wider the spread of dots for a feature, the more influential it is, and grid position's spread is by far the largest of the three.

**Top 3 Predictions (pre-qualifying):**
| Driver | Team | Podium Probability |
|--------|------|-------------------|
| HAM | Ferrari | 93.3% |
| LEC | Ferrari | 80.4% |
| VER | Red Bull Racing | 71.3% |
| NOR | McLaren | 69.6% |

## Model Performance
- **Best Model:** Random Forest (beat Logistic Regression and XGBoost)
- **ROC-AUC Score:** 0.98 (on 2023–2025 data)
- **Training Data:** 59 rows across 3 years of AUS GP results

## Features Used
| Feature | Description | Importance |
|---------|-------------|------------|
| GridPosition | Starting position on race day | Strongest |
| driver_aus_podium_rate | Driver's historical podium rate at AUS GP | Medium |
| constructor_aus_podium_rate | Constructor's historical podium rate at AUS GP | Weakest |

## Modeling Decisions
* The baseline model was Logistic Regression for binary classification (podium v. not podium). 
* Random Forest outperformed Logistic Regression. With only 59 rows, Logistic Regression's linear decision boundary didn't have enough complexity to capture the patterns that Random Forest's parallel tree approach could.
* Random Forest has a standard deviation that is almost half of the standard deviation of Logistic Regression. A lower standard deviation means the model performs more consistently across different folds.
* XGBoost builds trees sequentially where each tree learns from the mistakes of the previous one, making it
more powerful than Random Forest's parallel tree approach. 
* Only 3 out of ~20 drivers podium per race. Accuracy alone is misleading. A model predicting "no podium" for everyone would be 84.6% accurate but completely useless.
* ROC-AUC measures whether the model correctly ranks drivers by podium likelihood. It is a number that tells you how close your ranking is to perfect. 
* Raw Random Forest probabilities tend to be overconfident. Isotonic regression calibration was applied to ensure probability outputs are statistically reliable.

## Known Limitations
 * Due to a small dataset, we need to include more weightage for every row.
 * No DNF/reliability factor - the model has no way to know Red Bull had reliability issues in 2024. The model predicted VER as one of the starting grid, but did not account for the DNF/reliability factor.
 * We only have one circuit (AUS GP) data. The model has not seen any ther circuit, so it cannot generalize all of it. The dataset is entirely dependent on the Albert Park circuit. We only 3 years of data at this one circuit. 
 * We have no weather data/features. The dataset treats all 3 races equally with no weather context.
 * There are team name inconsistencies. They were changed and the model treats these as different constructors. So, these historical rates are not counted for.
 * New constructors (Audi, Cadillac) in 2026 and new drivers have no AUS history; they get constructor podium rate of 0.0 because they have no history in our dataset. 

## Repo Structure
```
F1_AUS_GP_Podium_Predictor/
├── 01_data_collection.ipynb        # data pull of 2023-2025 from AUS GP
├── 02_feature_engineering.ipynb    # Feature engineering and EDA
├── 03_baseline_model.ipynb         # Logistic regression and Random Forest
├── 04_xgboost_shap.ipynb           # XGBoost comparison, calibration, SHAP
├── 05_2026_predictions.ipynb       # 2026 driver lineup and predictions
├── aus_gp_data.csv                 # Raw cleaned race data
├── aus_gp_features.csv             # Engineered feature set
├── 2026_predictions.csv            # Final prediction outputs
├── calibrated_rf_model.pkl         # Trained calibrated model
├── shap_summary.png                # SHAP explainability plot
├── 2026_predictions.png            # Predictions visualization
├── requirements.txt                # Dependencies
└── README.md
```

## How to Run
1. Clone the repo
```
   git clone git@github.com:BK5102/F1-AUS-GP-Podium-Predictor.git
```
2. Install dependencies
```
   pip install -r requirements.txt
```
3. Run notebooks in order (01 → 05) in JupyterLab
4. To update predictions after qualifying, update GridPosition values in notebook 05 and rerun
   
## Post-Race Results

| Driver | Predicted Probability | Actual Finish |
|--------|----------------------|---------------|
| HAM | 93.3% | TBD |
| LEC | 80.4% | TBD |
| VER | 71.3% | TBD |
| NOR | 69.6% | TBD |

## Notes
* Raw model probabilities cannot be trusted. Probability Calibration relects these raw scores, so they represent what is actually happening.
* Probability calibration adjusts raw model scores that tend to be overconfident, pulling them toward more statistically reliable estimates.
* Isotonic regression calibration was applied because it is much more flexible and does not assume a particular shape. It finds the best adjustment.
* Adding the 2025 datadet with 20 extra rows and richer features values provided better distrbution plots. It had more variation resulting in a model that has more to learn from.
* There were no missing (null) values because the data is already well structured from the Fast F1 Python library. The 7 columns, driver number, abbreviation, team name, grid position, position and year, are always recorded for every driver in every race.
* The driver's podium rate calculation was performed by grouping all the rows together by each driver. Each group contained all the rows for that driver across 2023 and 2024. The constructor's podium rate has a similar interpretation.


## Connect
- LinkedIn: https://www.linkedin.com/in/bhavanakannan/ 
- GitHub: github.com/BK5102