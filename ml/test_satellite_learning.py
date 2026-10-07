"""
Exploratory check: how well do district features predict satellite-measured flooded area?
"""
import os
import json
import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score, f1_score

# Exploratory script (not the official benchmark): can district features predict the % of a
# district flooded (satellite-style extent), and what does a combined risk score look like?
# NOTE: it includes same-year columns (event_count, max_duration), so its numbers are optimistic.
def main():
    print("=" * 70)
    print("TESTING DATASET & PARAMETERS AGAINST SATELLITE INUNDATION TARGETS")
    print("=" * 70)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base_dir, 'data', 'processed', 'district_year_dataset_balanced_smote.csv')
    df = pd.read_csv(csv_path)
    print(f"Dataset Loaded: {len(df):,} balanced observations across 726 districts.")
    
    # Input columns (history, severity, water, rain features)
    features = [
        'event_count', 'max_duration', 'monsoon_peak_flag', 'cyclonic_flag',
        'y_lag1', 'prior_cum_events', 'prior_3yr_events', 'prior_5yr_events',
        'prior_10yr_events', 'flood_acceleration_rate', 'Population',
        'Parmanent_Water', 'Mean_Flood_Duration', 'dfsi_score',
        'drainage_stress_ratio', 'exposure_severity_index',
        'rainfall_anomaly_pct', 'hydro_rain_stress'
    ]
    
    X = df[features].fillna(0)
    y_sat_area = df['Corrected_Percent_Flooded_Area'].fillna(0)
    y_class = df['flood_occurred']
    
    # 1. Auxiliary Learning: Predict Satellite Inundation Severity
    print("\n--- 1. Auxiliary Learning Model: Predicting Satellite Flooded Area % ---")
    reg = XGBRegressor(n_estimators=80, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=2)
    
    # 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    r2_scores = []
    mae_scores = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], y_sat_area.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx], y_sat_area.iloc[val_idx]
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_val)
        r2 = r2_score(y_val, preds)
        mae = mean_absolute_error(y_val, preds)
        r2_scores.append(r2)
        mae_scores.append(mae)
    
    print(f"10-Fold CV R2 Score:   {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
    print(f"10-Fold CV MAE:        {np.mean(mae_scores):.4f}% flooded area")
    print("Note: hydro_rain_stress and exposure_severity_index are computed from the flooded-area value being predicted, so this R2 is inflated (a leakage check, not predictive skill).")
    
    # Feature Importances for Satellite Extent
    # Refit on all data; feature importance = share of the trees' splits/gain each column explains
    reg.fit(X, y_sat_area)
    importances = pd.Series(reg.feature_importances_, index=features).sort_values(ascending=False)
    print("\nTop Parameters Governing Satellite Inundation Extent:")
    for feat, imp in importances.head(6).items():
        print(f"  {feat:<26} Weight: {imp*100:.2f}%")
        
    # 2. Multi-Target Learning: Joint Inundation Severity Score
    print("\n--- 2. Output Optimization: Satellite-Calibrated Risk Output ---")
    clf = XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=2)
    clf.fit(X, y_class)
    prob = clf.predict_proba(X)[:, 1]
    
    # Multi-task combined score: Probability x Predicted Physical Severity
    # Predicted flooded area is a percentage, so keep it within 0-100
    pred_area = np.clip(reg.predict(X), 0, 100)
    # Same idea as IRS: probability x ln(1 + predicted flooded area %)
    calibrated_severity_index = prob * np.log1p(pred_area)
    
    print("Optimization Complete:")
    print(f"  Average Predicted Probability:       {prob.mean()*100:.1f}%")
    print(f"  Average Satellite Inundation Extent: {pred_area.mean():.2f}% of district")
    print(f"  Calibrated Severity Index Range:     {calibrated_severity_index.min():.3f} to {calibrated_severity_index.max():.3f}")

    # Save the results so 04_final_report.py can include them
    out = {
        'cv_r2_mean': float(np.mean(r2_scores)), 'cv_r2_std': float(np.std(r2_scores)),
        'cv_mae_mean': float(np.mean(mae_scores)),
        'top_features': {k: float(v) for k, v in importances.head(6).items()},
        'avg_predicted_probability_pct': float(prob.mean() * 100),
        'avg_predicted_flooded_area_pct': float(pred_area.mean()),
        'severity_index_min': float(calibrated_severity_index.min()),
        'severity_index_max': float(calibrated_severity_index.max()),
    }
    out_dir = os.path.join(base_dir, 'ml', 'models')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'satellite_learning_metrics.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {os.path.join(out_dir, 'satellite_learning_metrics.json')}")

if __name__ == '__main__':
    main()
