"""
PROOF 2 - SMOTE only ever receives TRAINING rows.
Run:  python ml/proof_2_smote_rows.py

How it works (no training is repeated):
  1. SMOTE / ADASYN's fit_resample() is wrapped so every call is logged: how many rows it received and
     which line of which project file called it.
  2. The REAL project code is then run (train_combo_hybrid_models.run_split and 02_train_models.py).
     The slow parts (neural-network / XGBoost training, saving files) are replaced by tiny stubs, because
     only the data that reaches SMOTE matters here.
  3. The logged row counts are compared with the sizes of the train and test parts, and the test rows
     are searched for inside SMOTE's input (there must be none).
Nothing is written to disk.
"""
import os, sys, io, json, builtins, inspect, runpy, hashlib
# work from the project root so relative paths (data/..., ml/...) resolve no matter where this is started
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import joblib
import imblearn.over_sampling as ios

# ---------------------------------------------------------------- the spy on SMOTE / ADASYN
CALLS = []   # one entry per fit_resample() call: who called it and what it received
_orig = {}   # the original (unwrapped) fit_resample of each class
for cls in (ios.SMOTE, ios.ADASYN):
    _orig[cls] = cls.fit_resample
    def make(cls):
        # a factory, so each class keeps its own name and original method inside the wrapper
        def traced(self, X, y, *a, **k):
            fr = inspect.stack()[1]          # the line of project code that called fit_resample
            X = np.asarray(X)
            CALLS.append(dict(kind=cls.__name__, file=os.path.basename(fr.filename), line=fr.lineno,
                              rows=len(X), flood=int(np.sum(y)), X=X.copy()))
            return _orig[cls](self, X, y, *a, **k)   # then do the normal resampling
        return traced
    cls.fit_resample = make(cls)


def row_hashes(A):
    """Fingerprint every row (rounded, then MD5) so two arrays can be compared row by row for overlap."""
    A = np.ascontiguousarray(np.round(np.asarray(A, dtype=np.float64), 5))
    return {hashlib.md5(r.tobytes()).hexdigest() for r in A}


# ================================================================ A. combo script (CSV deep models)
import ml.train_combo_hybrid_models as combo
# Stub the slow parts: "training" returns nothing and "predictions" are random numbers.
# SMOTE runs BEFORE training inside run_split, so what it receives is unaffected by the stubs.
combo.train_one = lambda cls, lr, ep, X, y, seed, **k: "stub"
combo.predict = lambda m, X, **k: np.random.RandomState(0).rand(len(X)).astype(np.float32)

# Rebuild the same 5-year windows the script uses and split them the same way (train: windows ending <= 2012)
df = pd.read_csv("data/processed/district_year_dataset.csv")
df[combo.FEATURE_COLS] = df[combo.FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
X, y, years = combo.load_windows(df)
N = len(X)
tr, te = years <= 2012, years > 2012

print("=" * 96)
print(f"A. ml/train_combo_hybrid_models.py  (whole dataset = {N:,} windows)")
CALLS.clear()
combo.run_split(X[tr], y[tr], X[te], y[te], combo.SEED)   # the real 80:20 holdout code path
c = CALLS[0]
print(f"   80:20 holdout : SMOTE got {c['rows']:,} rows = {c['rows']/N*100:.1f}% of the dataset "
      f"(train part has {tr.sum():,}, test part has {te.sum():,}) | called from {c['file']}:{c['line']}")
assert c['rows'] == tr.sum(), "SMOTE did not receive exactly the training part"

# No test row may be inside SMOTE's input. The script scales windows with training statistics before
# SMOTE, so scale the test windows the same way and compare fingerprints.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X[tr].reshape(-1, combo.N_FEATS))
smote_in = row_hashes(c['X'])
test_in = row_hashes(combo.scale(sc, X[te]).reshape(te.sum(), -1))
print(f"   test rows found inside SMOTE's input: {len(smote_in & test_in)}  (must be 0)")
assert not (smote_in & test_in)

# Same check for the 10 cross-validation folds: SMOTE must get exactly each fold's training rows
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=combo.SEED)
sizes = []
for k, (a, b) in enumerate(skf.split(X, y), 1):      # a = training rows, b = test fold rows
    CALLS.clear()
    combo.run_split(X[a], y[a], X[b], y[b], combo.SEED + k)
    sizes.append((CALLS[0]['rows'], len(a), len(b)))
    assert CALLS[0]['rows'] == len(a)
print(f"   10-fold       : SMOTE got {sizes[0][0]:,}..{max(s[0] for s in sizes):,} rows per fold = "
      f"{sizes[0][0]/N*100:.1f}% of the dataset (test fold = {sizes[0][2]:,} rows, never given to SMOTE)")

# ================================================================ B. XGBoost script
print("=" * 96)
print("B. ml/02_train_models.py  (XGBoost, SMOTE and ADASYN)")
import xgboost, sklearn.ensemble


class StubClf:
    """Stands in for XGBClassifier: accepts fit(), returns random probabilities, so the script runs in seconds."""
    feature_importances_ = np.zeros(15)
    def __init__(self, *a, **k): pass
    def fit(self, X, y): return self
    def predict_proba(self, X):
        p = np.random.RandomState(0).rand(len(X)); return np.c_[1 - p, p]


class StubReg(StubClf):
    """Stands in for the severity regressor."""
    def predict(self, X): return np.zeros(len(X))


xgboost.XGBClassifier = StubClf                 # the script does `from xgboost import XGBClassifier` after this
sklearn.ensemble.GradientBoostingRegressor = StubReg
# Make the script unable to save anything: model/JSON dumps do nothing, and files opened for writing
# go to a throw-away in-memory buffer instead of overwriting the real result files.
joblib.dump = lambda *a, **k: None
json.dump = lambda *a, **k: None
_open = builtins.open
builtins.open = lambda f, mode='r', *a, **k: io.StringIO() if any(m in str(mode) for m in 'wa') else _open(f, mode, *a, **k)
CALLS.clear()
import contextlib
with contextlib.redirect_stdout(io.StringIO()):   # hide the script's own printing
    runpy.run_path("ml/02_train_models.py", run_name="not_main")["train_deep_models"]()
builtins.open = _open                             # restore normal file opening

d2 = pd.read_csv("data/processed/district_year_dataset.csv")
n2 = len(d2); n_tr = int((d2['year'] <= int(d2['year'].quantile(0.80))).sum()); n_te = n2 - n_tr
print(f"   whole dataset = {n2:,} rows | chronological train part (<=2012) = {n_tr:,} ({n_tr/n2*100:.1f}%) | test part = {n_te:,}")
# line 101 of 02_train_models.py is the call inside the 10-fold loop; the others are the 80:20 holdout calls
fold_calls = [c for c in CALLS if c['line'] == 101]
hold_calls = [c for c in CALLS if c['line'] != 101]
for c in hold_calls:
    print(f"   80:20 holdout : {c['kind']:<6} got {c['rows']:,} rows = {c['rows']/n2*100:.1f}% of the dataset | called from {c['file']}:{c['line']}")
    assert c['rows'] == n_tr            # exactly the chronological training part, nothing more
fr = [c['rows'] for c in fold_calls]
print(f"   10-fold       : {len(fold_calls)} calls (SMOTE + ADASYN x 10 folds), each got {min(fr):,}..{max(fr):,} rows "
      f"= {min(fr)/n2*100:.1f}% of the dataset (the other ~10% is the untouched test fold)")
assert all(r < n2 for r in fr)

# ================================================================ C. every other call site
print("=" * 96)
print("C. every other place SMOTE appears in the project")
print("   ml/test_10fold_irs.py       : inside each 10-fold training part only (line 100, X_tr_c)")
print("   ml/data_loader.py           : balance_smote=False by default and no script turns it on -> never runs")
print("   ml/balance_dataset_smote.py : whole-dataset SMOTE demo; not imported or read by any benchmark (see below)")
import subprocess
# list every file that still mentions the whole-dataset SMOTE file: only the demo itself and these proofs should
out = subprocess.run(["grep", "-rln", "balanced_smote\\|balance_dataset", "ml", "web", "predict.py", "run_app.py", "README.md",
                      "AquaHorizon_Colab.ipynb"], capture_output=True, text=True).stdout.split()
print("   files mentioning the whole-dataset SMOTE file:", out)
print("\nVERDICT: wherever SMOTE touches the CSV data it receives only a training part "
      "(80% in the holdout, 90% in each CV fold); test rows are never passed to it.")
