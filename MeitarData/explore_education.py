import numpy as np
import pandas as pd
import yaml
from pathlib import Path

CFG_FILE = Path("NormativeHipModel.yml")
DATA_DIR = Path("../")

with open(CFG_FILE) as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(DATA_DIR / cfg["Dataset"]["file"])
df["age_open"] = pd.to_numeric(df["age_open"], errors="coerce")

flt = cfg["Filter"]
mask = (
    (df["diagnosis_mci_ad"] == flt["diagnosis"]) &
    (df["age_open"] >= flt["age_min"]) &
    (df["age_open"] <= flt["age_max"])
)
df = df[mask].copy()
df.rename(columns={"EstimatedTotalIntraCranialVol": "eTIV"}, inplace=True)

scale = cfg["Preprocessing"]["volume_scale"]
for col in ["eTIV", "BrainSegVolNotVent", "FS_L_Hippocampus", "FS_R_Hippocampus"]:
    df[col] /= scale

sex_enc = cfg["Sex"]["encoding"]
df["sex_enc"] = df["sex"].map(sex_enc)

# ── Two subsets: with and without education ───────────────────────────────────
df_all = df.dropna(subset=["age_open", "sex", "eTIV", "BrainSegVolNotVent",
                            "FS_L_Hippocampus", "FS_R_Hippocampus"])
df_edu = df.dropna(subset=["age_open", "sex", "eTIV", "BrainSegVolNotVent",
                            "FS_L_Hippocampus", "FS_R_Hippocampus", "dm15"])

print(f"Sessions with all vars (no edu): {len(df_all)}  ({df_all['SubjectID'].nunique()} subjects)")
print(f"Sessions with education:          {len(df_edu)}  ({df_edu['SubjectID'].nunique()} subjects)")

# Mean-center on the education subset (fair comparison within same N)
def center(data, cols):
    d = data.copy()
    for c in cols:
        d[c] -= d[c].mean()
    return d

df_edu = center(df_edu, ["age_open", "eTIV", "BrainSegVolNotVent",
                          "FS_L_Hippocampus", "FS_R_Hippocampus", "dm15"])

def fit_ols(X, y):
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ coef) ** 2))
    return coef, rss

def bic(rss, n, k):
    return n * np.log(rss / n) + k * np.log(n)

def r2(rss, y):
    return 1 - rss / np.sum((y - y.mean()) ** 2)

def build_X(data, preds):
    cols = ["sex_enc" if p == "sex" else p for p in preds]
    return np.column_stack([np.ones(len(data))] + [data[c].values for c in cols])

models = [
    ("eTIV  (no edu)",   ["age_open", "sex", "eTIV"]),
    ("eTIV  + edu",      ["age_open", "sex", "eTIV",              "dm15"]),
    ("Brain (no edu)",   ["age_open", "sex", "BrainSegVolNotVent"]),
    ("Brain + edu",      ["age_open", "sex", "BrainSegVolNotVent", "dm15"]),
]

targets = {"L": "FS_L_Hippocampus", "R": "FS_R_Hippocampus"}

print(f"\n{'Model':<22}  {'BIC_L':>10}  {'R²_L':>6}  {'BIC_R':>10}  {'R²_R':>6}  {'BIC_total':>10}")
print("-" * 74)

for label, preds in models:
    X   = build_X(df_edu, preds)
    n, k = len(df_edu), X.shape[1]
    bics, r2s = {}, {}
    for side, tcol in targets.items():
        y = df_edu[tcol].values
        coef, rss = fit_ols(X, y)
        bics[side] = bic(rss, n, k)
        r2s[side]  = r2(rss, y)
    total = bics["L"] + bics["R"]
    print(f"{label:<22}  {bics['L']:>10.2f}  {r2s['L']:>6.4f}  "
          f"{bics['R']:>10.2f}  {r2s['R']:>6.4f}  {total:>10.2f}")
