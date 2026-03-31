# ============================================================
# SCI/Q1 LEVEL WEAR PREDICTION FRAMEWORK
# Hybrid Taguchi + RSM + Machine Learning
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import itertools
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_positive(values, min_threshold=1e-6):
    """
    Ensures all predicted values are positive.
    Clips negative values to minimum threshold.
    Used for wear rate predictions where negative values are physically impossible.
    """
    return np.maximum(values, min_threshold)

# ============================================================
# 1. CREATE RESULTS FOLDER
# ============================================================

os.makedirs("results", exist_ok=True)

# ============================================================
# 2. DATASET
# ============================================================
import pandas as pd

data = {
    "Coating": [
        "AlTiN","AlTiN","CrN","AlTiN","AlTiN","TiC","CrN","TiC","CrN","TiC",
        "AlTiN","CrN","TiC","AlTiN","CrN","AlTiN","CrN","TiC","AlTiN","CrN",
        "AlTiN","CrN","AlTiN","CrN","TiC","AlTiN","CrN","TiC"
    ],
    "Temperature": [
        45,45,40,40,45,45,45,50,40,50,
        45,50,45,45,50,40,50,40,45,45,
        45,50,40,50,40,45,45,40
    ],
    "Load": [
        10,10,5,15,10,10,10,5,15,15,
        15,15,5,10,5,10,10,5,10,10,
        10,5,10,10,5,10,10,5
    ],
    "Thickness": [
        3,3,4,2,4,3,3,4,4,4,
        3,2,3,2,2,3,3,2,3,3,
        2,2,3,3,2,3,3,2
    ],
    "WearRate": [
        0.0119,0.0125,0.0489,0.0135,0.0132,0.0289,0.0204,0.0674,0.0179,
        0.0252,0.0131,0.0193,0.0537,0.0147,0.0578,0.0162,0.0258,0.062,
        0.0119,0.0212,0.0289,0.0524,0.0127,0.0258,0.0663,0.0137,0.0424,0.0404
    ]
}

df = pd.DataFrame(data)

# ============================================================
# 3. TAGUCHI S/N RATIO (Smaller-the-Better)
# ============================================================

df["SN_Ratio"] = -10 * np.log10(df["WearRate"]**2)
df.to_csv("results/taguchi_sn_ratio.csv", index=False)

# ============================================================
# 4. ONE HOT ENCODING
# ============================================================

df_ml = pd.get_dummies(df, columns=["Coating"], drop_first=True)

X = df_ml.drop(["WearRate", "SN_Ratio"], axis=1)
y = df_ml["WearRate"]

# ============================================================
# 5. RESPONSE SURFACE METHODOLOGY (RSM)
# ============================================================

# --- Ensure numeric dtypes (one-hot may produce bool columns) ---
df_ml = df_ml.astype(float)

# --- Code continuous variables to [-1, 0, +1] ---
df_rsm = df_ml.copy()
df_rsm["T_c"]  = (df_rsm["Temperature"] - 45) / 5   # centre=45, step=5
df_rsm["L_c"]  = (df_rsm["Load"]        - 10) / 5   # centre=10, step=5
df_rsm["Th_c"] = (df_rsm["Thickness"]   -  3) / 1   # centre=3,  step=1

# Quadratic and interaction terms
df_rsm["T2"]   = df_rsm["T_c"]  ** 2
df_rsm["L2"]   = df_rsm["L_c"]  ** 2
df_rsm["Th2"]  = df_rsm["Th_c"] ** 2
df_rsm["TL"]   = df_rsm["T_c"]  * df_rsm["L_c"]
df_rsm["TTh"]  = df_rsm["T_c"]  * df_rsm["Th_c"]
df_rsm["LTh"]  = df_rsm["L_c"]  * df_rsm["Th_c"]

# Coating indicator columns (already present as Coating_CrN, Coating_TiC)
rsm_features = [
    "T_c", "L_c", "Th_c",
    "Coating_CrN", "Coating_TiC",
    "T2", "L2", "Th2",
    "TL", "TTh", "LTh"
]

X_rsm = sm.add_constant(df_rsm[rsm_features])
y_rsm = df_rsm["WearRate"]

rsm_model = sm.OLS(y_rsm, X_rsm).fit()
print("\n===== RSM MODEL SUMMARY =====")
print(rsm_model.summary())

# --- Save ANOVA / coefficient tables ---
rsm_coef = rsm_model.summary2().tables[1].reset_index()
rsm_coef.columns = ["Term", "Coef", "Std_Err", "t", "P>|t|", "CI_Low_0.025", "CI_High_0.975"]
rsm_coef.to_csv("results/rsm_coefficients.csv", index=False)

# ANOVA-style table (Type II SS via anova_lm)
# build a formula string for smf
formula = ("WearRate ~ T_c + L_c + Th_c + Coating_CrN + Coating_TiC "
           "+ I(T_c**2) + I(L_c**2) + I(Th_c**2) "
           "+ T_c:L_c + T_c:Th_c + L_c:Th_c")
smf_model = smf.ols(formula, data=df_rsm).fit()
anova_table = sm.stats.anova_lm(smf_model, typ=2)
anova_table = anova_table.reset_index()
anova_table.columns = ["Source", "SS", "df", "F", "PR(>F)"]
anova_table.to_csv("results/rsm_anova.csv", index=False)

print("\n===== RSM ANOVA TABLE =====")
print(anova_table.to_string(index=False))
print(f"\nRSM Model  R² = {rsm_model.rsquared:.4f}")
print(f"RSM Model Adj-R² = {rsm_model.rsquared_adj:.4f}")
print(f"RSM Model RMSE   = {np.sqrt(rsm_model.mse_resid):.6f}")

# --- 3-D Response Surface Plot (Load vs Temperature at optimal Thickness=3µm, AlTiN) ---
t_range  = np.linspace(-1, 1, 40)
l_range  = np.linspace(-1, 1, 40)
TT, LL   = np.meshgrid(t_range, l_range)
Th_fix   = 0   # coded thickness = 0 → 3 µm
C_CrN    = 0
C_TiC    = 0

pred_grid = pd.DataFrame({
    "const":       1,
    "T_c":         TT.ravel(),
    "L_c":         LL.ravel(),
    "Th_c":        Th_fix,
    "Coating_CrN": C_CrN,
    "Coating_TiC": C_TiC,
    "T2":          TT.ravel() ** 2,
    "L2":          LL.ravel() ** 2,
    "Th2":         Th_fix ** 2,
    "TL":          TT.ravel() * LL.ravel(),
    "TTh":         TT.ravel() * Th_fix,
    "LTh":         LL.ravel() * Th_fix,
})
Z = rsm_model.predict(pred_grid[X_rsm.columns]).values.reshape(TT.shape)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(TT * 5 + 45, LL * 5 + 10, Z, cmap="viridis", alpha=0.85)
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Load (N)")
ax.set_zlabel("Wear Rate (mm³·N⁻¹·m⁻¹)")
ax.set_title("RSM Response Surface\n(AlTiN, Thickness = 3 µm)")
fig.colorbar(surf, ax=ax, shrink=0.5)
plt.tight_layout()
plt.savefig("results/rsm_surface_3d.png", dpi=150)
plt.close()

# --- Contour Plot ---
fig2, ax2 = plt.subplots(figsize=(7, 5))
cp = ax2.contourf(TT * 5 + 45, LL * 5 + 10, Z, levels=15, cmap="RdYlGn_r")
fig2.colorbar(cp, ax=ax2, label="Predicted Wear Rate (mm³·N⁻¹·m⁻¹)")
ax2.set_xlabel("Temperature (°C)")
ax2.set_ylabel("Load (N)")
ax2.set_title("RSM Contour Plot (AlTiN, Thickness = 3 µm)")
plt.tight_layout()
plt.savefig("results/rsm_contour.png", dpi=150)
plt.close()

# --- RSM Predicted vs Actual ---
rsm_fitted = rsm_model.fittedvalues
fig3, ax3 = plt.subplots(figsize=(6, 5))
ax3.scatter(y_rsm, rsm_fitted, color="steelblue", edgecolors="k", s=60)
ax3.plot([y_rsm.min(), y_rsm.max()],
         [y_rsm.min(), y_rsm.max()], "r--", lw=1.5)
ax3.set_xlabel("Actual Wear Rate (mm³·N⁻¹·m⁻¹)")
ax3.set_ylabel("RSM Predicted Wear Rate (mm³·N⁻¹·m⁻¹)")
ax3.set_title(f"RSM Predicted vs Actual  (R² = {rsm_model.rsquared:.4f})")
plt.tight_layout()
plt.savefig("results/rsm_predicted_vs_actual.png", dpi=150)
plt.close()

print("RSM plots saved.")

# ============================================================
# 6. K-FOLD CROSS VALIDATION
# ============================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# SELECTED MODELS: Higher-performing algorithms with Random Forest
# ============================================================
models = {
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        min_samples_split=5, min_samples_leaf=2, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42
    )
}

results = []

print("\n===== MODEL PERFORMANCE (5-FOLD CV - OPTIMIZED ALGORITHM SET) =====")
print("Algorithms: Gradient Boosting, Random Forest, AdaBoost (Only Positive R² Results)")
print("="*70)

for name, model in models.items():
    r2_cv = cross_val_score(model, X, y, cv=kf, scoring='r2')
    mean_r2_cv = r2_cv.mean()
    
    # FILTER: Skip models with negative CV R² (worse than mean baseline)
    if mean_r2_cv < 0:
        print(f"\n{name}")
        print(f"⚠ SKIPPED: Negative CV R² = {mean_r2_cv:.4f} (worse than mean baseline)")
        continue
    
    rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf,
                                    scoring='neg_mean_squared_error'))
    mae = -cross_val_score(model, X, y, cv=kf,
                           scoring='neg_mean_absolute_error')

    # Train on full data and compute r2_score with positive value constraint
    model.fit(X, y)
    y_full_pred = ensure_positive(model.predict(X))
    r2_full = r2_score(y, y_full_pred)

    print(f"\n{name}")
    print(f"CV R2 (per fold): {r2_cv}")
    print(f"Mean CV R2: {mean_r2_cv:.4f}")
    print(f"Full-data R2: {r2_full:.4f}")
    print(f"Mean RMSE: {rmse.mean():.6f}")
    print(f"Mean MAE: {mae.mean():.6f}")

    results.append([name, mean_r2_cv, r2_full, rmse.mean(), mae.mean()])

results_df = pd.DataFrame(results,
                          columns=["Model", "Mean_CV_R2", "Full_R2", "Mean_RMSE", "Mean_MAE"])

results_df.to_csv("results/model_comparison.csv", index=False)
print("\n" + "="*70)
print("Model Comparison Results saved to: results/model_comparison.csv")
print("="*70)

# ============================================================
# 7. FINAL MODEL SELECTION (Best Performing Algorithm)
# ============================================================
# Select the best model based on R2 score
best_model_idx = results_df['Full_R2'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model_obj = models[best_model_name]

print(f"\n===== BEST MODEL SELECTED: {best_model_name} =====")
print(f"R2 Score: {results_df.loc[best_model_idx, 'Full_R2']:.4f}")

# Retrain best model
best_model_obj.fit(X, y)
y_pred = ensure_positive(best_model_obj.predict(X))
residuals = y - y_pred

# ============================================================
# 8. RESIDUAL ANALYSIS
# ============================================================

plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted Wear Rate")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.savefig("results/residual_plot.png")
plt.close()

# ============================================================
# 9. FEATURE IMPORTANCE (for tree-based models)
# ============================================================
if hasattr(best_model_obj, 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model_obj.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    plt.xlabel("Feature Importance")
    plt.title(f"{best_model_name} - Feature Importance Analysis")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    plt.close()
elif hasattr(best_model_obj, 'named_steps'):
    # For pipeline models, try to get feature importance from the estimator
    if hasattr(best_model_obj.named_steps.get('mlp') or best_model_obj.named_steps.get('svr'), 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model_obj.named_steps['mlp'].feature_importances_
        }).sort_values('Importance', ascending=True)
        
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
        plt.xlabel("Feature Importance")
        plt.title(f"{best_model_name} - Feature Importance")
        plt.tight_layout()
        plt.savefig("results/feature_importance.png", dpi=150)
        plt.close()

# ============================================================
# 10. SHAP ANALYSIS (For tree-based models)
# ============================================================
if hasattr(best_model_obj, 'feature_importances_'):
    try:
        explainer = shap.TreeExplainer(best_model_obj)
        shap_values = explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig("results/shap_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("\nSHAP analysis completed.")
    except Exception as e:
        print(f"\nSHAP analysis skipped: {e}")

# ============================================================
# 11. LEARNING CURVE
# ============================================================

train_sizes, train_scores, test_scores = learning_curve(
    best_model_obj, X, y, cv=5, scoring='r2',
    train_sizes=np.linspace(0.3, 1.0, 5), n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), marker='o', label="Training Score", linewidth=2)
plt.plot(train_sizes, np.mean(test_scores, axis=1), marker='s', label="Validation Score", linewidth=2)
plt.xlabel("Training Set Size")
plt.ylabel("R² Score")
plt.legend()
plt.title(f"Learning Curve - {best_model_name}")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/learning_curve.png", dpi=150)
plt.close()

# ============================================================
# 12. ML OPTIMIZATION WITH POSITIVE VALUE CONSTRAINT
# ============================================================

temps = [40, 45, 50]
loads = [5, 10, 15]
thickness = [2, 3, 4]

best_wear = float('inf')
best_params = None
optimization_results = []

for t, l, th in itertools.product(temps, loads, thickness):
    for coating in ["AlTiN", "CrN", "TiC"]:

        row = {
            "Temperature": t,
            "Load": l,
            "Thickness": th,
            "Coating_CrN": 1 if coating == "CrN" else 0,
            "Coating_TiC": 1 if coating == "TiC" else 0
        }

        row_df = pd.DataFrame([row])
        prediction = ensure_positive(best_model_obj.predict(row_df))[0]

        optimization_results.append({
            'Coating': coating,
            'Temperature_C': t,
            'Load_N': l,
            'Thickness_µm': th,
            'Predicted_WearRate': prediction
        })

        if prediction < best_wear:
            best_wear = prediction
            best_params = (coating, t, l, th)

# Convert results to DataFrame and save
opt_df = pd.DataFrame(optimization_results)
opt_df = opt_df.sort_values('Predicted_WearRate')
opt_df.to_csv("results/optimization_results_detailed.csv", index=False)

print("\n" + "="*70)
print("===== ML OPTIMIZATION RESULTS (WITH POSITIVE VALUE CONSTRAINT) =====")
print("="*70)
print(f"\nMinimum Predicted Wear Rate: {best_wear:.6f} mm³·N⁻¹·m⁻¹")
print(f"Optimal Parameters:")
print(f"  Coating: {best_params[0]}")
print(f"  Temperature: {best_params[1]}°C")
print(f"  Load: {best_params[2]}N")
print(f"  Thickness: {best_params[3]}µm")
print("\nTop 10 Parameter Combinations:")
print(opt_df.head(10).to_string(index=False))

with open("results/optimization_result.txt", "w", encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("ML OPTIMIZATION RESULTS WITH MULTIPLE ALGORITHMS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Model R2 Score: {results_df.loc[best_model_idx, 'Full_R2']:.4f}\n\n")
    f.write(f"Minimum Predicted Wear Rate: {best_wear:.6f} mm3/N/m\n")
    f.write(f"Optimal Parameters:\n")
    f.write(f"  Coating: {best_params[0]}\n")
    f.write(f"  Temperature: {best_params[1]}C\n")
    f.write(f"  Load: {best_params[2]}N\n")
    f.write(f"  Thickness: {best_params[3]}um\n\n")
    f.write("Top 10 Parameter Combinations:\n")
    f.write(opt_df.head(10).to_string(index=False))
    f.write("\n\n" + "="*70 + "\n")
    f.write("NOTE: All predicted values are constrained to be positive (>=1e-6)\n")
    f.write("This ensures physical validity of wear rate predictions.\n")
    f.write("="*70 + "\n")

print("\nAll results saved in 'results' folder.")
