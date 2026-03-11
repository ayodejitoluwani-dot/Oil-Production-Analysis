import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

BLUE  = "#1F4E79"; LIGHT = "#2E75B6"; ACCENT = "#F4A21E"
GREEN = "#27AE60"; RED   = "#C0392B"; GRAY   = "#555555"; BG = "#F7F9FC"

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'figure.facecolor': BG,
    'axes.facecolor': BG, 'axes.spines.top': False, 'axes.spines.right': False
})

df = pd.read_csv('d/Users/chugga/Desktop/Seplat Production Analysis/seplat_production.csv')

# ── CHART 1: Production Overview ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Production Overview — Seplat Well Portfolio', fontsize=15, fontweight='bold', color=BLUE)

# Avg production by field
field_prod = df.groupby('field')['production_bopd'].mean().sort_values(ascending=False)
colors = [BLUE, LIGHT, ACCENT, GREEN]
bars = axes[0].bar(field_prod.index, field_prod.values, color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, field_prod.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{val:.0f}', ha='center', fontsize=10, fontweight='bold', color=BLUE)
axes[0].set_title('Avg Production by Field (bopd)', fontweight='bold', color=BLUE)
axes[0].set_ylabel('Barrels of Oil Per Day')
axes[0].set_ylim(0, field_prod.max() * 1.2)

# Production by year
year_prod = df.groupby('year')['production_bopd'].mean()
axes[1].plot(year_prod.index, year_prod.values, color=BLUE, lw=2.5, marker='o', markersize=7)
axes[1].fill_between(year_prod.index, year_prod.values, alpha=0.15, color=BLUE)
axes[1].set_title('Avg Production Trend by Year', fontweight='bold', color=BLUE)
axes[1].set_ylabel('Barrels of Oil Per Day')
axes[1].set_xlabel('Year')

# Production distribution
axes[2].hist(df['production_bopd'], bins=30, color=LIGHT, edgecolor='white', linewidth=0.8)
axes[2].axvline(df['production_bopd'].mean(), color=RED, lw=2, linestyle='--', label=f'Mean: {df["production_bopd"].mean():.0f} bopd')
axes[2].set_title('Production Distribution', fontweight='bold', color=BLUE)
axes[2].set_xlabel('Production (bopd)')
axes[2].set_ylabel('Number of Records')
axes[2].legend()

plt.tight_layout()
plt.savefig('d/Users/chugga/Desktop/Seplat Production Analysis/oil_chart1_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 1 saved")

# ── CHART 2: Key Production Drivers ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Key Drivers of Oil Production', fontsize=15, fontweight='bold', color=BLUE)

# Reservoir pressure vs production
axes[0,0].scatter(df['reservoir_pressure_psi'], df['production_bopd'],
                  alpha=0.4, color=BLUE, s=20)
m, b = np.polyfit(df['reservoir_pressure_psi'], df['production_bopd'], 1)
x_line = np.linspace(df['reservoir_pressure_psi'].min(), df['reservoir_pressure_psi'].max(), 100)
axes[0,0].plot(x_line, m*x_line+b, color=RED, lw=2)
axes[0,0].set_title('Reservoir Pressure vs Production', fontweight='bold', color=BLUE)
axes[0,0].set_xlabel('Reservoir Pressure (psi)'); axes[0,0].set_ylabel('Production (bopd)')

# Water cut vs production
axes[0,1].scatter(df['water_cut_pct'], df['production_bopd'],
                  alpha=0.4, color=ACCENT, s=20)
m2, b2 = np.polyfit(df['water_cut_pct'], df['production_bopd'], 1)
x2 = np.linspace(df['water_cut_pct'].min(), df['water_cut_pct'].max(), 100)
axes[0,1].plot(x2, m2*x2+b2, color=RED, lw=2)
axes[0,1].set_title('Water Cut % vs Production', fontweight='bold', color=BLUE)
axes[0,1].set_xlabel('Water Cut (%)'); axes[0,1].set_ylabel('Production (bopd)')

# Well age vs production
age_prod = df.groupby('well_age_years')['production_bopd'].mean()
axes[1,0].plot(age_prod.index, age_prod.values, color=GREEN, lw=2.5, marker='o', markersize=5)
axes[1,0].fill_between(age_prod.index, age_prod.values, alpha=0.15, color=GREEN)
axes[1,0].set_title('Well Age vs Avg Production', fontweight='bold', color=BLUE)
axes[1,0].set_xlabel('Well Age (Years)'); axes[1,0].set_ylabel('Avg Production (bopd)')

# Downtime vs production
axes[1,1].scatter(df['downtime_hours'], df['production_bopd'],
                  alpha=0.4, color=RED, s=20)
m3, b3 = np.polyfit(df['downtime_hours'], df['production_bopd'], 1)
x3 = np.linspace(df['downtime_hours'].min(), df['downtime_hours'].max(), 100)
axes[1,1].plot(x3, m3*x3+b3, color=BLUE, lw=2)
axes[1,1].set_title('Downtime Hours vs Production', fontweight='bold', color=BLUE)
axes[1,1].set_xlabel('Downtime (Hours/Month)'); axes[1,1].set_ylabel('Production (bopd)')

plt.tight_layout()
plt.savefig('d/Users/chugga/Desktop/Seplat Production Analysis/oil_chart2_drivers.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 2 saved")

# ── ML MODELS ────────────────────────────────────────────────────────────────
features = ['well_age_years', 'reservoir_pressure_psi', 'water_cut_pct',
            'gas_oil_ratio', 'choke_size_64ths', 'downtime_hours',
            'rainfall_mm', 'pipeline_utilisation_pct']

X = df[features]
y = df['production_bopd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_s, y_train)
lr_pred = lr.predict(X_test_s)
lr_mae  = mean_absolute_error(y_test, lr_pred)
lr_r2   = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae  = mean_absolute_error(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_mae  = mean_absolute_error(y_test, gb_pred)
gb_r2   = r2_score(y_test, gb_pred)
gb_rmse = np.sqrt(mean_squared_error(y_test, gb_pred))

print(f"\nLinear Regression  — MAE: {lr_mae:.1f} bopd, R²: {lr_r2:.3f}, RMSE: {lr_rmse:.1f}")
print(f"Random Forest      — MAE: {rf_mae:.1f} bopd, R²: {rf_r2:.3f}, RMSE: {rf_rmse:.1f}")
print(f"Gradient Boosting  — MAE: {gb_mae:.1f} bopd, R²: {gb_r2:.3f}, RMSE: {gb_rmse:.1f}")

# ── CHART 3: Model Performance ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Model Performance Comparison', fontsize=15, fontweight='bold', color=BLUE)

# R² comparison
models = ['Linear\nRegression', 'Random\nForest', 'Gradient\nBoosting']
r2s    = [lr_r2, rf_r2, gb_r2]
maes   = [lr_mae, rf_mae, gb_mae]
bar_colors = [LIGHT, ACCENT, GREEN]
bars = axes[0].bar(models, r2s, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, r2s):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')
axes[0].set_title('R² Score (higher = better)', fontweight='bold', color=BLUE)
axes[0].set_ylim(0, 1.1)
axes[0].axhline(0.9, color=RED, lw=1.5, linestyle='--', alpha=0.5, label='0.90 threshold')
axes[0].legend(fontsize=9)

# MAE comparison
bars2 = axes[1].bar(models, maes, color=bar_colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, maes):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
axes[1].set_title('Mean Absolute Error — bopd\n(lower = better)', fontweight='bold', color=BLUE)
axes[1].set_ylabel('MAE (bopd)')

# Best model: actual vs predicted
best_pred = gb_pred  # gradient boosting is best
axes[2].scatter(y_test, best_pred, alpha=0.4, color=GREEN, s=25)
min_v, max_v = y_test.min(), y_test.max()
axes[2].plot([min_v, max_v], [min_v, max_v], color=RED, lw=2, linestyle='--', label='Perfect prediction')
axes[2].set_title('Gradient Boosting:\nActual vs Predicted (bopd)', fontweight='bold', color=BLUE)
axes[2].set_xlabel('Actual Production (bopd)')
axes[2].set_ylabel('Predicted Production (bopd)')
axes[2].legend()

plt.tight_layout()
plt.savefig('d/Users/chugga/Desktop/Seplat Production Analysis/oil_chart3_models.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 3 saved")

# ── CHART 4: Feature Importance + Revenue Impact ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Strategic Insights', fontsize=15, fontweight='bold', color=BLUE)

# Feature importance
importances = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=True)
feat_colors = [RED if v == importances.max() else LIGHT for v in importances.values]
axes[0].barh(importances.index, importances.values, color=feat_colors, edgecolor='white')
axes[0].set_title('Top Production Drivers\n(Gradient Boosting)', fontweight='bold', color=BLUE)
axes[0].set_xlabel('Importance Score')

# Revenue impact of downtime
oil_price_usd = 85  # approx Brent crude
avg_prod = df['production_bopd'].mean()
downtime_scenarios = [0, 24, 48, 72, 120, 168, 240]
revenue_loss = [(h / 24) * avg_prod * oil_price_usd for h in downtime_scenarios]
axes[1].bar([f'{h}h' for h in downtime_scenarios], revenue_loss,
            color=[GREEN, LIGHT, LIGHT, ACCENT, ACCENT, RED, RED], edgecolor='white')
axes[1].set_title('Estimated Revenue Loss per Well\nby Monthly Downtime (@$85/bbl)', fontweight='bold', color=BLUE)
axes[1].set_xlabel('Monthly Downtime')
axes[1].set_ylabel('Revenue Loss (USD)')
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.savefig('d/Users/chugga/Desktop/Seplat Production Analysis/oil_chart4_insights.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart 4 saved")

# Save metrics
import json
metrics = {
    "total_records": len(df), "wells": int(df['well_id'].nunique()),
    "avg_production": round(float(df['production_bopd'].mean()), 1),
    "lr_r2": round(lr_r2, 3), "lr_mae": round(lr_mae, 1),
    "rf_r2": round(rf_r2, 3), "rf_mae": round(rf_mae, 1),
    "gb_r2": round(gb_r2, 3), "gb_mae": round(gb_mae, 1),
    "top_driver": importances.index[-1],
    "downtime_24h_loss_usd": round(revenue_loss[1], 0),
    "downtime_168h_loss_usd": round(revenue_loss[5], 0),
}
with open('d/Users/chugga/Desktop/Seplat Production Analysis/oil_metrics.json', 'w') as f:
    json.dump(metrics, f)
print("\nAll done.")
print(json.dumps(metrics, indent=2))
