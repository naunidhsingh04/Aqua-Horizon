import numpy as np

class KFoldEnsembleClassifier:
    """Soft-voting ensemble of XGBoost models trained across K-Fold cross-validation partitions."""
    def __init__(self, models):
        self.models = models
        # Feature importance of the ensemble = average of each fold model's importance
        self.feature_importances_ = np.mean([m.feature_importances_ for m in models], axis=0)

    def predict_proba(self, X):
        # Soft voting: final probability = average of the 10 models' probabilities
        probs = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return probs

    def predict(self, X, threshold=0.40):
        # Flood (1) if averaged P(flood) >= 0.40; lower than 0.5 favours catching more floods
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


if __name__ == '__main__':
    # Self-test: three small XGBoost models on synthetic data, then their soft-voting average.
    from xgboost import XGBClassifier
    rng = np.random.RandomState(42)
    X = rng.randn(600, 6)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.randn(600) > 0.4).astype(int)
    models = [XGBClassifier(n_estimators=40, max_depth=3, random_state=s, eval_metric='logloss').fit(X[::2], y[::2])
              for s in (1, 2, 3)]
    ens = KFoldEnsembleClassifier(models)
    X_test, y_test = X[1::2], y[1::2]
    print("=" * 66)
    print("K-FOLD SOFT-VOTING ENSEMBLE (self-test on synthetic data)")
    print("=" * 66)
    for i, m in enumerate(models, 1):
        acc = (m.predict(X_test) == y_test).mean() * 100
        print(f"  Member model {i}: accuracy {acc:.1f}%")
    p = ens.predict_proba(X_test)[:, 1]
    print(f"  Ensemble (average of the 3 probabilities), threshold 0.50: accuracy {(ens.predict(X_test, threshold=0.5) == y_test).mean() * 100:.1f}%")
    print(f"  First 5 test rows -> averaged flood probability: {np.round(p[:5], 3)}")
    print(f"  Ensemble feature importances (average of members): {np.round(ens.feature_importances_, 3)}")
