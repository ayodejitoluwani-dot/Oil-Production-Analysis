# Oil Production Forecasting — Seplat Energy Case Study

A machine learning project forecasting monthly oil production across a portfolio of 20 Nigerian onshore wells, identifying key production drivers and delivering strategic recommendations for operations and planning teams.

## Problem Statement
Accurate production forecasting directly impacts revenue guidance, capital allocation, and offtake agreements. This project answers: *given what we know about a well today, how much will it produce next month?*

## Dataset
- 1,000 monthly production records across 20 wells
- 4 fields: Sapele, Oben, Eleme, Otiaseki (2018–2023)
- Features include reservoir pressure, water cut, well age, downtime, choke size, and rainfall
- Average production: 724 bopd

## Models Built
| Model | R² Score | MAE (bopd) |
|-------|----------|------------|
| Gradient Boosting | 0.946 | 73.5 |
| Random Forest | 0.939 | 77.3 |
| Linear Regression | 0.897 | 100.9 |

✅ Gradient Boosting selected as best model (R²: 0.946)

## Key Findings
- Reservoir pressure is the strongest predictor of production output
- Water cut above 60% reduces production by up to 50%
- Wells aged 8–12 years show the steepest production decline
- Every 24 hours of unplanned downtime costs ~$61,500 per well at $85/bbl

## Business Recommendations
1. Deploy monthly well production scorecard using the forecast model
2. Prioritise pressure maintenance on mid-life wells (8–14 years)
3. Implement water cut early warning system
4. Track and reduce unplanned downtime across the portfolio

## Tools & Libraries
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- Jupyter / VS Code

## Files
| File | Description |
|------|-------------|
| `seplat_production_analysis.py` | Full analysis and ML pipeline |
| `seplat_production.csv` | Simulated well production dataset |

## Author
**Ayodeji Toluwani**
AI / Data Intern — NCAIR
[LinkedIn] | ayodejitoluwani@gmail.com
