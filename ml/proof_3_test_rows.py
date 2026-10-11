"""
Run:  python ml/proof_3_test_rows.py
Runs the REAL project code and proves, for the held-out 20% test part:
  (1) the rows the model predicts on are byte-identical to the original real rows (not resampled, not synthetic),
  (2) their flood / non-flood ratio is the original natural one (not balanced),
  (3) the labels fed to accuracy / recall / precision / F1 / ROC-AUC are exactly the true labels of those rows.
Neural-net / XGBoost training is stubbed out and nothing is written to disk.
"""
import os, sys, io, json, builtins, runpy, contextlib
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import joblib
import sklearn.metrics as skm, sklearn.ensemble, xgboost

share = lambda y: f"{int(np.sum(y)):,} flood / {int((np.asarray(y) == 0).sum()):,} non-flood = {np.mean(y)*100:.1f}% flood"

# ================================================================= A. combo script (CSV deep models)
import ml.train_combo_hybrid_models as combo
PRED, SCORE = [], []
combo.train_one = lambda cls, lr, ep, X, y, seed, **k: "stub"
def fake_predict(m, X, **k):
    PRED.append(np.array(X, copy=True)); return np.random.RandomState(0).rand(len(X)).astype(np.float32)
combo.predict = fake_predict
_score = combo.score
def spy_score(y, p, thr):
    SCORE.append((np.array(y, copy=True), len(p))); return _score(y, p, thr)
combo.score = spy_score

df = pd.read_csv("data/processed/district_year_dataset.csv")
df[combo.FEATURE_COLS] = df[combo.FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)
X, y, years = combo.load_windows(df)
tr, te = years <= 2012, years > 2012
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X[tr].reshape(-1, combo.N_FEATS))

print("=" * 98)
print("A. ml/train_combo_hybrid_models.py - 80:20 holdout")
print(f"   original test part (real rows, windows ending 2013-2023): {te.sum():,} rows | {share(y[te])}")
PRED.clear(); SCORE.clear()
combo.run_split(X[tr], y[tr], X[te], y[te], combo.SEED)
p_test = PRED[0]                                  # what the model was asked to predict on
expected = combo.scale(sc, X[te])                 # the real test windows, scaled with TRAIN statistics only
print(f"   model predicted on        : {len(p_test):,} rows -> identical to the original test rows: {np.array_equal(p_test, expected)}")
print(f"   (a SMOTE-balanced test set would have {int((y[te]==0).sum())*2:,} rows; the real one has {te.sum():,})")
assert np.array_equal(p_test, expected)
labels = [s[0] for s in SCORE]
ok = all(np.array_equal(l, y[te]) for l in labels)
print(f"   metrics scored against    : {len(SCORE)} scorings (7 models), every one on the true labels of those {len(labels[0]):,} rows: {ok}")
print(f"   labels used for scoring   : {share(labels[0])}  (natural ratio, unbalanced)")
assert ok

print("   10-fold CV test folds:")
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=combo.SEED)
all_ok = True; tested = 0
for k, (a, b) in enumerate(skf.split(X, y), 1):
    PRED.clear(); SCORE.clear()
    combo.run_split(X[a], y[a], X[b], y[b], combo.SEED + k)
    sc_k = StandardScaler().fit(X[a].reshape(-1, combo.N_FEATS))
    same_x = np.array_equal(PRED[0], combo.scale(sc_k, X[b]))
    same_y = all(np.array_equal(s[0], y[b]) for s in SCORE)
    all_ok &= same_x and same_y; tested += len(b)
    if k in (1, 10):
        print(f"     fold {k:<2}: test fold {len(b):,} rows | {share(y[b])} | rows identical: {same_x} | labels identical: {same_y}")
print(f"   all 10 folds: rows identical & labels identical = {all_ok} | the 10 test folds together cover {tested:,} of {len(X):,} rows, each row tested exactly once")
assert all_ok and tested == len(X)

# ================================================================= B. 02_train_models.py
print("=" * 98)
print("B. ml/02_train_models.py - chronological 80:20 holdout (XGBoost)")
XP, YT = [], []
class StubClf:
    feature_importances_ = np.zeros(15)
    def __init__(self, *a, **k): pass
    def fit(self, X, y): return self
    def predict_proba(self, X):
        XP.append(np.array(X, copy=True)); p = np.random.RandomState(0).rand(len(X)); return np.c_[1 - p, p]
class StubReg(StubClf):
    def predict(self, X): return np.zeros(len(X))
xgboost.XGBClassifier = StubClf
sklearn.ensemble.GradientBoostingRegressor = StubReg
_acc = skm.accuracy_score
def spy_acc(yt, yp, *a, **k):
    YT.append(np.array(yt, copy=True)); return _acc(yt, yp, *a, **k)
skm.accuracy_score = spy_acc
joblib.dump = lambda *a, **k: None
json.dump = lambda *a, **k: None
_open = builtins.open
builtins.open = lambda f, mode='r', *a, **k: io.StringIO() if any(m in str(mode) for m in 'wa') else _open(f, mode, *a, **k)
with contextlib.redirect_stdout(io.StringIO()):
    runpy.run_path("ml/02_train_models.py", run_name="not_main")["train_deep_models"]()
builtins.open = _open

d = pd.read_csv("data/processed/district_year_dataset.csv")
cols = ['prior_cum_events','prior_3yr_events','prior_5yr_events','prior_10yr_events','flood_acceleration_rate','dfsi_score',
        'dfsi_rank','Corrected_Percent_Flooded_Area','Parmanent_Water','Mean_Flood_Duration','Population',
        'drainage_stress_ratio','exposure_severity_index','hydro_rain_stress','rainfall_anomaly_pct']
d[cols] = d[cols].fillna(0).replace([np.inf, -np.inf], 0)
cut = int(d['year'].quantile(0.80)); t = d[d['year'] > cut]
Xt, yt = t[cols].values, t['flood_occurred'].values
print(f"   original test part (real rows, years {cut+1}-{int(d['year'].max())}): {len(t):,} rows | {share(yt)}")
names = ["Baseline", "SMOTE", "ADASYN"]
for i, nm in enumerate(names):
    xp, y_ = XP[-3 + i], YT[-3 + i]
    ix, iy = np.array_equal(xp, Xt), np.array_equal(y_, yt)
    print(f"   {nm:<9}: predicted on {len(xp):,} rows (identical to original test rows: {ix}) | metrics scored on {len(y_):,} true labels (identical: {iy})")
    assert ix and iy
print("\nVERDICT: the 20% test part is passed to the models and to every metric exactly as it was in the real data - same rows, same")
print("         labels, same natural unbalanced ratio. It is used only for testing, never for fitting, resampling or scaling.")
