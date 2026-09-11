import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN

def train_deep_models():
    print("=" * 75)
    print("DEEP ML MODEL TRAINING & BENCHMARKING (80:20 CHRONOLOGICAL SPLIT)")
    print("=" * 75)
    
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

    # Strict 80:20 Chronological Split
    cutoff_year = int(df['year'].quantile(0.80)) # 2012
    train_mask = df['year'] <= cutoff_year
    test_mask = df['year'] > cutoff_year

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values

    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values

    print(f"Training Partition 80% (1967-{cutoff_year}): {len(df_train)} records (Pos: {y_train.sum()}, Neg: {len(y_train)-y_train.sum()})")
    print(f"Testing Partition  20% ({cutoff_year+1}-2023): {len(df_test)} records (Pos: {y_test.sum()}, Neg: {len(y_test)-y_test.sum()}) [UNTOUCHED]")

    # Resampling on Training Split Only
    print("\nApplying Resampling on Training Data Only...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    adasyn = ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

    configs = {
        "Baseline (Control - Raw)": (X_train, y_train),
        "SMOTE Balanced": (X_train_smote, y_train_smote),
        "ADASYN Balanced": (X_train_adasyn, y_train_adasyn)
    }

    results = {}
    trained_models = {}

    for name, (X_tr, y_tr) in configs.items():
        print(f"\nTraining Deep XGBoost Model: {name}...")
        clf = XGBClassifier(
            n_estimators=160,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            random_state=42,
            eval_metric='logloss'
        )
        clf.fit(X_tr, y_tr)
        trained_models[name] = clf

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.40).astype(int)

        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        pr_auc = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "PR-AUC": round(pr_auc, 4),
            "ROC-AUC": round(roc_auc, 4)
        }
        print(f"Results [{name}] -> Recall: {rec*100:.2f}%, Precision: {prec*100:.2f}%, F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}")

    print("\n" + "=" * 75)
    print("FINAL SCIENTIFIC BENCHMARK SUMMARY TABLE:")
    print("=" * 75)
    print(pd.DataFrame(results).T.to_string())

    # Champion model selection
    champion_name = "ADASYN Balanced"
    champion_clf = trained_models[champion_name]

    # Secondary Severity Regressor
    print("\nTraining Secondary Hydrological Severity Regressor...")
    df_flood = df_train[df_train[target_col] == 1].copy()
    y_sev = np.log1p(df_flood['max_duration'] * np.log1p(df_flood['dfsi_score']))
    X_sev = df_flood[feature_cols].values

    reg_sev = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    reg_sev.fit(X_sev, y_sev)

    # Save artifacts
    os.makedirs('ml/models', exist_ok=True)
    joblib.dump(champion_clf, 'ml/models/champion_classifier.joblib')
    joblib.dump(reg_sev, 'ml/models/severity_regressor.joblib')
    
    with open('ml/models/benchmark_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    importances = dict(zip(feature_cols, [round(float(x), 4) for x in champion_clf.feature_importances_]))
    with open('ml/models/feature_importance.json', 'w') as f:
        json.dump(importances, f, indent=2)
    print("\nTop Feature Importances:", sorted(importances.items(), key=lambda x: x[1], reverse=True)[:6])

if __name__ == '__main__':
    train_deep_models()
