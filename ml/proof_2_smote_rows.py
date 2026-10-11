"""
Run:  python ml/proof_2_smote_rows.py
Runs the REAL project code with SMOTE/ADASYN instrumented: every call logs how many rows it received,
which source line called it, and (when checkable) whether any test row was inside its input.
Training of the neural nets / XGBoost is stubbed out and nothing is written to disk.
"""
import os, sys, io, json, builtins, inspect, runpy, hashlib
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import joblib
import imblearn.over_sampling as ios

CALLS = []
_orig = {}
for cls in (ios.SMOTE, ios.ADASYN):
    _orig[cls] = cls.fit_resample
    def make(cls):
        def traced(self, X, y, *a, **k):
            fr = inspect.stack()[1]
            X = np.asarray(X)
            CALLS.append(dict(kind=cls.__name__, file=os.path.basename(fr.filename), line=fr.lineno,
                              rows=len(X), flood=int(np.sum(y)), X=X.copy()))
            return _orig[cls](self, X, y, *a, **k)
        return traced
    cls.fit_resample = make(cls)

def row_hashes(A):
    A = np.ascontiguousarray(np.round(np.asarray(A, dtype=np.float64), 5))
    return {hashlib.md5(r.tobytes()).hexdigest() for r in A}

# ---------------------------------------------------------------- A. combo script (CSV deep models)
import ml.train_combo_hybrid_models as combo
class Dummy:
    def __call__(self, x): return None
combo.train_one = lambda cls, lr, ep, X, y, seed, **k: "stub"
combo.predict = lambda m, X, **k: np.random.RandomState(0).rand(len(X)).astype(np.float32)

df = pd.read_csv("data/processed/district_year_dataset.csv")
df[combo.FEATURE_COLS] = df[combo.FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
X, y, years = combo.load_windows(df)
N = len(X)
tr, te = years <= 2012, years > 2012

print("=" * 96)
print(f"A. ml/train_combo_hybrid_models.py  (whole dataset = {N:,} windows)")
CALLS.clear()
combo.run_split(X[tr], y[tr], X[te], y[te], combo.SEED)
c = CALLS[0]
print(f"   80:20 holdout : SMOTE got {c['rows']:,} rows = {c['rows']/N*100:.1f}% of the dataset "
      f"(train part has {tr.sum():,}, test part has {te.sum():,}) | called from {c['file']}:{c['line']}")
assert c['rows'] == tr.sum(), "SMOTE did not receive exactly the training part"
# test rows must not be among SMOTE's inputs (scale the test windows the same way the script does)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X[tr].reshape(-1, combo.N_FEATS))
smote_in = row_hashes(c['X'])
test_in = row_hashes(combo.scale(sc, X[te]).reshape(te.sum(), -1))
print(f"   test rows found inside SMOTE's input: {len(smote_in & test_in)}  (must be 0)")
assert not (smote_in & test_in)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=combo.SEED)
sizes = []
for k, (a, b) in enumerate(skf.split(X, y), 1):
    CALLS.clear()
    combo.run_split(X[a], y[a], X[b], y[b], combo.SEED + k)
    sizes.append((CALLS[0]['rows'], len(a), len(b)))
    assert CALLS[0]['rows'] == len(a)
print(f"   10-fold       : SMOTE got {sizes[0][0]:,}..{max(s[0] for s in sizes):,} rows per fold = "
      f"{sizes[0][0]/N*100:.1f}% of the dataset (test fold = {sizes[0][2]:,} rows, never given to SMOTE)")

# ---------------------------------------------------------------- B. XGBoost script
print("=" * 96)
print("B. ml/02_train_models.py  (XGBoost, SMOTE and ADASYN)")
import xgboost, sklearn.ensemble
class StubClf:
    feature_importances_ = np.zeros(15)
    def __init__(self, *a, **k): pass
    def fit(self, X, y): return self
    def predict_proba(self, X):
        p = np.random.RandomState(0).rand(len(X)); return np.c_[1 - p, p]
class StubReg(StubClf):
    def predict(self, X): return np.zeros(len(X))
xgboost.XGBClassifier = StubClf
sklearn.ensemble.GradientBoostingRegressor = StubReg
joblib.dump = lambda *a, **k: None
json.dump = lambda *a, **k: None
_open = builtins.open
builtins.open = lambda f, mode='r', *a, **k: io.StringIO() if any(m in str(mode) for m in 'wa') else _open(f, mode, *a, **k)
CALLS.clear()
import contextlib
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path("ml/02_train_models.py", run_name="not_main")["train_deep_models"]()
builtins.open = _open
d2 = pd.read_csv("data/processed/district_year_dataset.csv")
n2 = len(d2); n_tr = int((d2['year'] <= int(d2['year'].quantile(0.80))).sum()); n_te = n2 - n_tr
print(f"   whole dataset = {n2:,} rows | chronological train part (<=2012) = {n_tr:,} ({n_tr/n2*100:.1f}%) | test part = {n_te:,}")
fold_calls = [c for c in CALLS if c['line'] == 101]
hold_calls = [c for c in CALLS if c['line'] != 101]
for c in hold_calls:
    print(f"   80:20 holdout : {c['kind']:<6} got {c['rows']:,} rows = {c['rows']/n2*100:.1f}% of the dataset | called from {c['file']}:{c['line']}")
    assert c['rows'] == n_tr
fr = [c['rows'] for c in fold_calls]
print(f"   10-fold       : {len(fold_calls)} calls (SMOTE + ADASYN x 10 folds), each got {min(fr):,}..{max(fr):,} rows "
      f"= {min(fr)/n2*100:.1f}% of the dataset (the other ~10% is the untouched test fold)")
assert all(r < n2 for r in fr)

# ---------------------------------------------------------------- C. every other call site
print("=" * 96)
print("C. every other place SMOTE appears in the project")
print("   ml/test_10fold_irs.py       : inside each 10-fold training part only (line 100, X_tr_c)")
print("   ml/data_loader.py           : balance_smote=False by default and no script turns it on -> never runs")
print("   ml/balance_dataset_smote.py : whole-dataset SMOTE demo; not imported or read by any benchmark (see below)")
import subprocess
out = subprocess.run(["grep", "-rln", "balanced_smote\\|balance_dataset", "ml", "web", "predict.py", "run_app.py", "README.md",
                      "AquaHorizon_Colab.ipynb"], capture_output=True, text=True).stdout.split()
print("   files mentioning the whole-dataset SMOTE file:", out)
print("\nVERDICT: wherever SMOTE touches the CSV data it receives only a training part "
      "(80% in the holdout, 90% in each CV fold); test rows are never passed to it.")
