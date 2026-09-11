import numpy as np

class KFoldEnsembleClassifier:
    """Soft-voting ensemble of XGBoost models trained across K-Fold cross-validation partitions."""
    def __init__(self, models):
        self.models = models
        self.feature_importances_ = np.mean([m.feature_importances_ for m in models], axis=0)

    def predict_proba(self, X):
        probs = np.mean([m.predict_proba(X) for m in self.models], axis=0)
        return probs

    def predict(self, X, threshold=0.40):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)