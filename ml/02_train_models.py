import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from ml.ensemble import KFoldEnsembleClassifier

def train_deep_models():
    print("=" * 80)
    print("AQUA HORIZON — 10-FOLD CROSS-VALIDATION & BENCHMARKING ENGINE")
    print("=" * 80)
    
    data_path = 'data/processed/district_year_dataset.csv'
    df = pd.read_csv(data_path)

    feature_cols = [
        'prior_cum_events',
        'prior_3yr_events',
        'prior_5yr_events',
        'prior_10yr_events',
        'flood_acceleration_rate',
        'dfsi_score',
        'dfsi_rank',
        'Corrected_Percent_Flooded_Area',
        'Parmanent_Water',
        'Mean_Flood_Duration',
        'Population',
        'drainage_stress_ratio',
        'exposure_severity_index',
        'hydro_rain_stress',
        'rainfall_anomaly_pct'
    ]
    target_col = 'flood_occurred'

    df[feature_cols] = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    X = df[feature_cols].values
    y = df[target_col].values

    # =========================================================================
    # 1. 10-FOLD STRATIFIED CROSS-VALIDATION PIPELINE
    # =========================================================================
    print("\n[PHASE 1] Executing 10-Fold Stratified Cross-Validation with ADASYN Resampling...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    kfold_models = []
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Resample strictly within training fold (Zero Leakage)
        adasyn = ADASYN(random_state=42, n_neighbors=5)
        X_tr_res, y_tr_res = adasyn.fit_resample(X_tr, y_tr)

        clf = XGBClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            random_state=42 + fold,
            eval_metric='logloss',
            n_jobs=-1
        )
        clf.fit(X_tr_res, y_tr_res)
        kfold_models.append(clf)

        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.40).astype(int)

        rec = recall_score(y_val, preds)
        prec = precision_score(y_val, preds)
        f1 = f1_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)
        prauc = average_precision_score(y_val, probs)

        fold_results.append({
            'Fold': f'Fold {fold+1}',
            'Recall': rec,
            'Precision': prec,
            'F1-Score': f1,
            'ROC-AUC': auc,
            'PR-AUC': prauc
        })
        print(f"  Fold {fold+1}/10 -> Recall: {rec*100:.2f}%, Precision: {prec*100:.2f}%, F1: {f1:.4f}, ROC-AUC: {auc:.4f}, PR-AUC: {prauc:.4f}")

    df_folds = pd.DataFrame(fold_results)
    cv_mean_rec = float(df_folds['Recall'].mean())
    cv_mean_prec = float(df_folds['Precision'].mean())
    cv_mean_f1 = float(df_folds['F1-Score'].mean())
    cv_mean_auc = float(df_folds['ROC-AUC'].mean())
    cv_mean_prauc = float(df_folds['PR-AUC'].mean())

    print("\n--- 10-Fold Stratified Cross-Validation Summary ---")
    print(f"  Mean Test Recall:    {cv_mean_rec*100:.2f}% (Std: +/- {df_folds['Recall'].std()*100:.2f}%)")
    print(f"  Mean Test Precision: {cv_mean_prec*100:.2f}% (Std: +/- {df_folds['Precision'].std()*100:.2f}%)")
    print(f"  Mean F1-Score:       {cv_mean_f1:.4f}")
    print(f"  Mean ROC-AUC:        {cv_mean_auc:.4f}")
    print(f"  Mean PR-AUC:         {cv_mean_prauc:.4f}")

    # =========================================================================
    # 2. CHRONOLOGICAL 80:20 RATIO SPLIT (1967-2012 vs 2013-2023)
    # =========================================================================
    print("\n[PHASE 2] Benchmarking Chronological 80:20 Split (1967-2012 vs 2013-2023)...")
    cutoff_year = int(df['year'].quantile(0.80)) # 2012
    train_mask = df['year'] <= cutoff_year
    test_mask = df['year'] > cutoff_year

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    X_train, y_train = df_train[feature_cols].values, df_train[target_col].values
    X_test, y_test = df_test[feature_cols].values, df_test[target_col].values

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    adasyn_holdout = ADASYN(random_state=42, n_neighbors=5)
    X_train_adasyn, y_train_adasyn = adasyn_holdout.fit_resample(X_train, y_train)

    configs = {
        "Baseline (Control - Raw)": (X_train, y_train),
        "SMOTE Balanced": (X_train_smote, y_train_smote),
        "ADASYN Balanced": (X_train_adasyn, y_train_adasyn)
    }

    results = {}
    for name, (X_tr, y_tr) in configs.items():
        clf = XGBClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.40).astype(int)

        results[name] = {
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
        }

    # Add 10-Fold Cross-Validation Benchmark to official results
    results["10-Fold Stratified CV (ADASYN + XGBoost)"] = {
        "Precision": round(cv_mean_prec, 4),
        "Recall": round(cv_mean_rec, 4),
        "F1-Score": round(cv_mean_f1, 4),
        "PR-AUC": round(cv_mean_prauc, 4),
        "ROC-AUC": round(cv_mean_auc, 4)
    }

    print("\n" + "=" * 80)
    print("OFFICIAL BENCHMARK COMPARISON TABLE (HOLDOUT & 10-FOLD CV):")
    print("=" * 80)
    print(pd.DataFrame(results).T.to_string())

    # =========================================================================
    # 3. BUILD CROSS-VALIDATED ENSEMBLE CHAMPION MODEL
    # =========================================================================
    print("\n[PHASE 3] Constructing 10-Fold Cross-Validated Soft-Voting Ensemble Champion...")
    ensemble_champion = KFoldEnsembleClassifier(kfold_models)

    # Secondary Severity Regressor
    print("Training Secondary Hydrological Severity Regressor...")
    df_flood = df_train[df_train[target_col] == 1].copy()
    y_sev = np.log1p(df_flood['max_duration'] * np.log1p(df_flood['dfsi_score']))
    X_sev = df_flood[feature_cols].values

    reg_sev = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    reg_sev.fit(X_sev, y_sev)

    # Save artifacts
    os.makedirs('ml/models', exist_ok=True)
    joblib.dump(ensemble_champion, 'ml/models/champion_classifier.joblib')
    joblib.dump(reg_sev, 'ml/models/severity_regressor.joblib')
    
    with open('ml/models/benchmark_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    importances = dict(zip(feature_cols, [round(float(x), 4) for x in ensemble_champion.feature_importances_]))
    with open('ml/models/feature_importance.json', 'w') as f:
        json.dump(importances, f, indent=2)

    print(f"\nSuccessfully exported Champion 10-Fold Model to ml/models/champion_classifier.joblib")
    print("Top Feature Importances:", sorted(importances.items(), key=lambda x: x[1], reverse=True)[:6])

if __name__ == '__main__':
    train_deep_models()
