"""
PROOF 4 - the GAN only ever sees flood chips from the 80% training part of the image data.
Run:  python ml/proof_4_gan_inputs.py
Runs the REAL ml/image_pipeline_india.py main() with the GAN training and the classifier training stubbed
(speed) and every call logged: what the GAN was trained on, what was augmented, and what the classifier
was tested on. Nothing is written to disk.

How it works: train_sar_dcgan, augment_with_gan and train_and_evaluate_model are replaced by spies that keep a
copy of their inputs. After the real main() has run, those copies are compared with the chips of the 80% train
part and the 20% test part, which are rebuilt here independently with the same seed.
"""
import os, sys, io, json, builtins, hashlib, contextlib
# work from the project root so relative paths resolve no matter where this is started
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.getcwd())
import numpy as np
import ml.image_pipeline_india as ip
from sklearn.model_selection import train_test_split, StratifiedKFold

# H: fingerprint (MD5) of every chip, so chips can be compared for overlap without comparing pixels one by one
H = lambda A: [hashlib.md5(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest() for a in A]
# one-line summary of a label array: how many flood / non-flood and the flood share
share = lambda y: f"{int(np.sum(y)):,} flood / {int((np.asarray(y) == 0).sum()):,} non-flood = {np.mean(y)*100:.1f}% flood"

# one entry per call, in the order main() makes them:
#   GAN  = the chips each GAN was trained on
#   AUG  = what was augmented and how many synthetic chips were added
#   EVAL = (train X, train y, test X, test y) given to the classifier
GAN, AUG, EVAL = [], [], []


def fake_gan(chips, epochs=12, latent_dim=64, batch_size=32):
    # spy: remember the exact chips the GAN would have been trained on; skip the real (slow) GAN training
    GAN.append(np.array(chips, copy=True))
    return ip.SARGenerator(latent_dim=latent_dim, channels=2).to(ip.device).eval()   # untrained: only used to make noise images
_aug = ip.augment_with_gan


def spy_aug(gen, X, y, n_synthetic=120, latent_dim=64):
    # spy: run the REAL augmentation, but keep a copy of its input and output (original part vs synthetic part)
    aX, ay = _aug(gen, X, y, n_synthetic=n_synthetic, latent_dim=latent_dim)
    AUG.append((len(X), np.array(y, copy=True), len(aX), n_synthetic, np.array(aX[:len(X)], copy=True), np.array(aX[len(X):], copy=True)))
    return aX, ay


def fake_eval(Xt, yt, Xv, yv, **k):
    # spy: remember what the classifier is trained and tested on; return dummy scores instead of training
    EVAL.append((np.array(Xt, copy=True), np.array(yt, copy=True), np.array(Xv, copy=True), np.array(yv, copy=True)))
    return dict(accuracy=0., precision=0., recall=0., f1_score=0., iou=0., roc_auc=0., train_acc=0., gen_gap=0.)
ip.train_sar_dcgan, ip.augment_with_gan, ip.train_and_evaluate_model = fake_gan, spy_aug, fake_eval

# make main() unable to save anything: JSON dumps do nothing, files opened for writing go to a throw-away buffer
json.dump = lambda *a, **k: None
_open = builtins.open
builtins.open = lambda f, mode='r', *a, **k: io.StringIO() if any(m in str(mode) for m in 'wa') else _open(f, mode, *a, **k)
with contextlib.redirect_stdout(io.StringIO()):   # hide main()'s own printing
    ip.main()                                     # the REAL pipeline: split, CLAHE, GAN, 80:20 run, 10-fold run
builtins.open = _open                             # restore normal file opening

# ---- ground truth, rebuilt independently of main()
with contextlib.redirect_stdout(io.StringIO()):
    raw, masks, labels = ip.load_sen1floods11_india_chips(patch_size=64)
    clahe = ip.apply_clahe_sar(raw)
N = len(labels)
# same stratified 80:20 split (same seed) as image_pipeline_india.main(): tr = 80% train part, te = 20% test part
tr, te = train_test_split(np.arange(N), test_size=0.20, random_state=ip.SEED, stratify=labels)
# hs: the set of chip fingerprints, used to check "no overlap" / "exactly these chips"
hs = lambda A: set(H(A))

print("=" * 98)
print(f"IMAGE DATASET: {N:,} real Sentinel-1 chips | {share(labels)}")
print(f"80:20 split  : train part {len(tr):,} chips ({len(tr)/N*100:.0f}%) | test part {len(te):,} chips ({len(te)/N*100:.0f}%)")
print("=" * 98)
print("80:20 HOLDOUT (experiment C - the GAN run)")
# the first GAN call in main() is the 80:20 holdout GAN (experiment C)
chips = GAN[0]
train_flood = clahe[tr][labels[tr] == 1]
print(f"  GAN trained on          : {len(chips)} chips")
print(f"  flood chips of the 80%  : {len(train_flood)} -> GAN input is exactly those chips: {hs(chips) == hs(train_flood)}")
print(f"  test-part chips in GAN  : {len(hs(chips) & hs(clahe[te]))} of {len(te)}  (must be 0)")
print(f"  non-flood chips in GAN  : {int((np.isin(H(chips), H(clahe[labels == 0]))).sum())}  (must be 0: the GAN only ever sees flood chips)")
assert hs(chips) == hs(train_flood) and not (hs(chips) & hs(clahe[te]))
n0, ytr_, n1, nsyn, kept, synth = AUG[0]
print(f"  augmentation input      : {n0} chips (the 80% train part) -> {n1} after adding {nsyn} synthetic flood chips")
print(f"  original 960 kept as-is : {np.array_equal(kept, clahe[tr])} | new chips are all labelled flood")
assert n0 == len(tr) and np.array_equal(kept, clahe[tr])
Xt, yt, Xv, yv = EVAL[2]    # classifier calls in order: A (raw SAR), B (CLAHE), C (CLAHE + GAN) -> index 2 is the GAN run
print(f"  classifier trained on   : {len(Xt)} chips ({len(tr)} real + {len(Xt)-len(tr)} synthetic) | {share(yt)}")
print(f"  classifier TESTED on    : {len(Xv)} chips, identical to the original untouched test chips: {np.array_equal(Xv, clahe[te])} "
      f"| labels identical: {np.array_equal(yv, labels[te])} | {share(yv)}")
assert np.array_equal(Xv, clahe[te]) and np.array_equal(yv, labels[te])
assert not (hs(Xv) & hs(Xt)), "test chip also present in training data"
print(f"  test chips found in the GAN-augmented training data: {len(hs(Xv) & hs(Xt))}  (must be 0)")

print("\n10-FOLD CV (a fresh GAN per fold)")
# the 10 cross-validation folds, rebuilt with the same seed; the calls after the holdout belong to the folds in order
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=ip.SEED)
fold_gan = GAN[1:]; fold_aug = AUG[1:]; fold_eval = EVAL[3:]
assert len(fold_gan) == len(fold_aug) == len(fold_eval) == 10
for k, (a, b) in enumerate(skf.split(clahe, labels)):
    ch, ev = fold_gan[k], fold_eval[k]
    fl = clahe[a][labels[a] == 1]   # the flood chips of this fold's training part: all the GAN may see
    # GAN input must be exactly those chips, contain no test-fold chip, and the classifier must be tested on the untouched test fold
    ok = hs(ch) == hs(fl) and not (hs(ch) & hs(clahe[b])) and np.array_equal(ev[2], clahe[b]) and np.array_equal(ev[3], labels[b])
    assert ok
    if k in (0, 9):
        print(f"  fold {k+1:<2}: GAN trained on {len(ch)} flood chips of the {len(a):,}-chip training part | test-fold chips in GAN: "
              f"{len(hs(ch) & hs(clahe[b]))} | classifier tested on {len(ev[2])} untouched real chips ({share(ev[3])})")
print("  all 10 folds: GAN input = flood chips of that fold's training part only; the test fold never reached the GAN or the training data: True")

print("\nVERDICT: the GAN is trained only on flood chips from the 80% training part and its synthetic chips are added to that part only.")
print("         The 20% test chips (and each CV test fold) are never given to the GAN, never augmented, and are what the metrics are computed on.")
