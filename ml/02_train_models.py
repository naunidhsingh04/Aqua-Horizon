import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
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

    def make_clf(seed):
        return XGBClassifier(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            random_state=seed,
            eval_metric='logloss',
            n_jobs=-1
        )

    STRATEGIES = {
        "Baseline (Control - Raw)": ("No Resampling", None),
        "SMOTE Balanced": ("SMOTE", SMOTE),
        "ADASYN Balanced": ("ADASYN", ADASYN)
    }

    # =========================================================================
    # 1. 10-FOLD STRATIFIED CROSS-VALIDATION PIPELINE (every resampling strategy)
    # =========================================================================
    print("\n[PHASE 1] Executing 10-Fold Stratified Cross-Validation for every resampling strategy...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    kfold_summaries = {}
    champion_fold_models = []  # ADASYN fold models power the deployed ensemble

    for name, (strategy_label, resampler_cls) in STRATEGIES.items():
        print(f"\n  -- {name} ({strategy_label}) --")
        fold_results = []
        fold_models = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Resample strictly within the training fold only (zero leakage)
            if resampler_cls is not None:
                resampler = resampler_cls(random_state=42, n_neighbors=5) if resampler_cls is ADASYN else resampler_cls(random_state=42)
                X_tr_res, y_tr_res = resampler.fit_resample(X_tr, y_tr)
            else:
                X_tr_res, y_tr_res = X_tr, y_tr

            clf = make_clf(seed=42 + fold)
            clf.fit(X_tr_res, y_tr_res)
            fold_models.append(clf)

            probs = clf.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.40).astype(int)

            fold_results.append({
                'Accuracy': accuracy_score(y_val, preds),
                'Recall': recall_score(y_val, preds),
                'Precision': precision_score(y_val, preds),
                'F1-Score': f1_score(y_val, preds),
                'ROC-AUC': roc_auc_score(y_val, probs),
                'PR-AUC': average_precision_score(y_val, probs)
            })

        df_folds = pd.DataFrame(fold_results)
        kfold_summaries[name] = {
            metric: {'mean': float(df_folds[metric].mean()), 'std': float(df_folds[metric].std())}
            for metric in ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'ROC-AUC', 'PR-AUC']
        }
        print(f"    10-Fold Mean Accuracy: {kfold_summaries[name]['Accuracy']['mean']*100:.2f}% "
              f"(+/- {kfold_summaries[name]['Accuracy']['std']*100:.2f}%) | "
              f"Recall: {kfold_summaries[name]['Recall']['mean']*100:.2f}% | "
              f"ROC-AUC: {kfold_summaries[name]['ROC-AUC']['mean']:.4f}")

        if name == "ADASYN Balanced":
            champion_fold_models = fold_models

    # =========================================================================
    # 2. CHRONOLOGICAL 80:20 RATIO SPLIT (1967-2012 vs 2013-2023) — single-split baseline
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

    holdout_configs = {
        "Baseline (Control - Raw)": (X_train, y_train),
        "SMOTE Balanced": (X_train_smote, y_train_smote),
        "ADASYN Balanced": (X_train_adasyn, y_train_adasyn)
    }

    results = {}
    for name, (X_tr, y_tr) in holdout_configs.items():
        clf = make_clf(seed=42)
        clf.fit(X_tr, y_tr)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.40).astype(int)

        strategy_label = STRATEGIES[name][0]
        kf = kfold_summaries[name]

        results[name] = {
            "strategy": strategy_label,
            "holdout": {
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "F1-Score": round(f1_score(y_test, y_pred), 4),
                "PR-AUC": round(average_precision_score(y_test, y_prob), 4),
                "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
            },
            "kfold": {
                "Accuracy": round(kf['Accuracy']['mean'], 4),
                "Accuracy_std": round(kf['Accuracy']['std'], 4),
                "Precision": round(kf['Precision']['mean'], 4),
                "Recall": round(kf['Recall']['mean'], 4),
                "Recall_std": round(kf['Recall']['std'], 4),
                "F1-Score": round(kf['F1-Score']['mean'], 4),
                "PR-AUC": round(kf['PR-AUC']['mean'], 4),
                "ROC-AUC": round(kf['ROC-AUC']['mean'], 4)
            }
        }

    print("\n" + "=" * 80)
    print("OFFICIAL BENCHMARK COMPARISON TABLE (HOLDOUT & 10-FOLD CV, EVERY STRATEGY):")
    print("=" * 80)
    for name, r in results.items():
        print(f"  {name}: Holdout Acc {r['holdout']['Accuracy']*100:.1f}% | "
              f"10-Fold Acc {r['kfold']['Accuracy']*100:.1f}% (+/- {r['kfold']['Accuracy_std']*100:.1f}%)")

    # =========================================================================
    # 3. BUILD CROSS-VALIDATED ENSEMBLE CHAMPION MODEL
    # =========================================================================
    print("\n[PHASE 3] Constructing 10-Fold Cross-Validated Soft-Voting Ensemble Champion (ADASYN)...")
    ensemble_champion = KFoldEnsembleClassifier(champion_fold_models)

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
