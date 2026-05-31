# ============================================================
# LiAgent — ML Training Script
# Ionic Conductivity Prediction for Solid State Electrolytes
# 5 Algorithms: XGBoost, Gradient Boosting, LightGBM,
#               Random Forest, Neural Network
# Dataset: 4,407 datapoints | 342 compounds | 146 features
# ============================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import (train_test_split,
                                      cross_val_score, KFold)
from sklearn.ensemble import (RandomForestRegressor,
                               GradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────
PROJECT_DIR = Path(r"C:\Users\raalnuba\Desktop\Ionic_Conductivity_ML")

# ── Load dataset ──────────────────────────────────────────────
df = pd.read_excel(PROJECT_DIR / 'merged_dataset.xlsx')
fc = joblib.load(PROJECT_DIR / 'merged_feature_cols.pkl')

print(f"Dataset   : {len(df)} rows")
print(f"Compounds : {df['material'].nunique()} unique")
print(f"Features  : {len(fc)+1} (145 Magpie + Temperature)")

# ── Features and target ───────────────────────────────────────
X = df[fc + ['Temp']].fillna(0)
y = df['cond(log)']

# ── 80/20 train/test split ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining  : {len(X_train)} rows (80%)")
print(f"Testing   : {len(X_test)} rows (20%)")

# ── Define 5 ML models ────────────────────────────────────────
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=300,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1),

    'XGBoost': XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0),

    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42),

    'LightGBM': LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1),

    'Neural Network': Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            learning_rate_init=0.001,
            solver='adam'))]),
}

# ── 5-Fold Cross Validation + Test Evaluation ─────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'='*95}")
print(f"{'Random Split 5-Fold CV Results':^95}")
print(f"{'='*95}")
print(f"{'Model':<22} {'CV R²':>7} {'CV Std':>7} "
      f"{'Test R²':>8} {'MAE':>7} {'RMSE':>7}  "
      f"Fold R² (1-5)")
print(f"{'-'*95}")

results       = {}
trained_models = {}

for name, model in models.items():
    # 5-fold CV on training set
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=kf, scoring='r2', n_jobs=-1)

    # Train on full training set
    model.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = model.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(np.mean((y_test - y_pred)**2))

    results[name] = {
        'cv_r2'  : cv_scores.mean(),
        'cv_std' : cv_scores.std(),
        'test_r2': r2,
        'mae'    : mae,
        'rmse'   : rmse,
        'folds'  : cv_scores,
    }
    trained_models[name] = model

    fold_str = ' | '.join([f"{s:.3f}" for s in cv_scores])
    print(f"  {name:<20} {cv_scores.mean():>7.3f} "
          f"{cv_scores.std():>7.3f} {r2:>8.3f} "
          f"{mae:>7.3f} {rmse:>7.3f}  {fold_str}")

# ── Save results to Excel ─────────────────────────────────────
rows = []
for name, res in results.items():
    fold_str = ' | '.join([f"{s:.3f}" for s in res['folds']])
    rows.append({
        'Model'         : name,
        'CV R²'         : round(res['cv_r2'], 4),
        'CV Std'        : round(res['cv_std'], 4),
        'Test R²'       : round(res['test_r2'], 4),
        'MAE'           : round(res['mae'], 4),
        'Test RMSE'     : round(res['rmse'], 4),
        'Fold R² (1-5)' : fold_str,
    })

df_results = pd.DataFrame(rows)
df_results.to_excel(
    PROJECT_DIR / 'ml_results.xlsx', index=False)
print(f"\nResults saved: ml_results.xlsx")

# ── Identify and save best model ──────────────────────────────
best_name = max(results,
                key=lambda x: results[x]['test_r2'])
print(f"\nBest model: {best_name} "
      f"(Test R²={results[best_name]['test_r2']:.4f})")

# Save all models
joblib.dump(trained_models['XGBoost'],
            PROJECT_DIR / 'merged_model.pkl')
joblib.dump(trained_models['Gradient Boosting'],
            PROJECT_DIR / 'merged_model_GB_backup.pkl')
joblib.dump(trained_models['LightGBM'],
            PROJECT_DIR / 'lgbm_model.pkl')
joblib.dump(trained_models['Neural Network'],
            PROJECT_DIR / 'mlp_model.pkl')
joblib.dump(trained_models['Random Forest'],
            PROJECT_DIR / 'new_model.pkl')
print(f"All 5 models saved successfully!")
