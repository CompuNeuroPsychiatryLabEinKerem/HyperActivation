"""
Normative hippocampal volume model — stepwise BIC-based selection.
Runs two parallel model families (eTIV-based and Brain-based) and writes
both to NormativeHipModel.yml together with shared offsets.
One observation per subject (last session). Complete cases on all predictors.
"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from collections import Counter

CFG_FILE = Path("NormativeHipModel.yml")
DATA_DIR = Path("../")

# ── Load config ───────────────────────────────────────────────────────────────
with open(CFG_FILE) as f:
    cfg = yaml.safe_load(f)

# ── Load & filter ─────────────────────────────────────────────────────────────
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

df["sex_enc"] = df["sex"].map(cfg["Sex"]["encoding"])

required = ["age_open", "sex_enc", "dm15", "eTIV", "BrainSegVolNotVent",
            "FS_L_Hippocampus", "FS_R_Hippocampus"]
df = df.dropna(subset=required)
df = df.sort_values("SessionNum").groupby("SubjectID").last().reset_index()

print(f"Dataset: {len(df)} subjects (last session, complete cases)")
print(df["sex"].value_counts().to_string())

# ── Mean-center; store offsets ────────────────────────────────────────────────
center_cols = ["age_open", "eTIV", "BrainSegVolNotVent", "dm15",
               "FS_L_Hippocampus", "FS_R_Hippocampus"]
offsets = {col: float(round(df[col].mean(), 4)) for col in center_cols}
for col in center_cols:
    df[col] -= offsets[col]

print("\nOffsets (means):")
for k, v in offsets.items():
    print(f"  {k}: {v}")

# ── Term helpers ──────────────────────────────────────────────────────────────
NO_POLY   = {"sex_enc"}
MAX_POWER = 3

def term_values(term, data):
    v = np.ones(len(data))
    for f in term:
        v = v * data[f].values
    return v

def term_label(term, display):
    c = Counter(term)
    parts = [display[f] if n == 1 else f"{display[f]}^{n}" for f, n in c.items()]
    return " * ".join(parts)

def features_present(terms):
    return set(f for t in terms for f in t)

def generate_candidates(current_terms, base_features):
    current_set = set(current_terms)
    present     = features_present(current_terms)
    candidates  = set()
    for feat in base_features:
        if feat not in present:
            candidates.add((feat,))
        else:
            if feat not in NO_POLY:
                for p in range(1, MAX_POWER):
                    tp, tp1 = tuple([feat] * p), tuple([feat] * (p + 1))
                    if tp in current_set and tp1 not in current_set:
                        candidates.add(tp1)
                        break
            for other in present:
                if other != feat:
                    interaction = tuple(sorted([feat, other]))
                    if interaction not in current_set:
                        candidates.add(interaction)
    return candidates

# ── OLS / BIC ─────────────────────────────────────────────────────────────────
def build_X(terms, data):
    return np.column_stack([np.ones(len(data))] +
                           [term_values(t, data) for t in terms])

def fit_ols(X, y):
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    rss = float(np.sum((y - X @ coef) ** 2))
    return coef, rss

def compute_bic(rss, n, k):
    return n * np.log(rss / n) + k * np.log(n)

def evaluate(terms, data):
    X         = build_X(terms, data)
    n, k      = len(data), X.shape[1]
    y_l, y_r  = data["FS_L_Hippocampus"].values, data["FS_R_Hippocampus"].values
    cl, rss_l = fit_ols(X, y_l)
    cr, rss_r = fit_ols(X, y_r)
    return {
        "bic_total": compute_bic(rss_l, n, k) + compute_bic(rss_r, n, k),
        "bic_l": compute_bic(rss_l, n, k), "bic_r": compute_bic(rss_r, n, k),
        "r2_l":  1 - rss_l / np.sum((y_l - y_l.mean()) ** 2),
        "r2_r":  1 - rss_r / np.sum((y_r - y_r.mean()) ** 2),
        "coef_l": cl, "coef_r": cr,
    }

# ── Stepwise runner ───────────────────────────────────────────────────────────
def run_stepwise(start_terms, base_features, display, label):
    current_terms = list(start_terms)
    res           = evaluate(current_terms, df)

    print(f"\n{'=' * 70}")
    start_str = "Intercept + " + " + ".join(term_label(t, display) for t in start_terms)
    print(f"STEPWISE — {label}  (start: {start_str})")
    print(f"{'=' * 70}")
    print(f"\nStart:  BIC={res['bic_total']:.2f}  "
          f"R²_L={res['r2_l']:.4f}  R²_R={res['r2_r']:.4f}")

    for step in range(1, 20):
        candidates = generate_candidates(current_terms, base_features)
        if not candidates:
            print("\nNo more candidates — stopping.")
            break

        print(f"\n── Step {step}  ({len(candidates)} candidates) "
              + "─" * (44 - len(str(len(candidates)))))

        rows = sorted(
            [(evaluate(current_terms + [c], df), c) for c in candidates],
            key=lambda x: x[0]["bic_total"]
        )
        best_res, best_cand = rows[0]

        for r, cand in rows:
            marker = "  <--" if (cand == best_cand
                                 and r["bic_total"] < res["bic_total"]) else ""
            print(f"  +{term_label(cand, display):<24} BIC={r['bic_total']:8.2f}  "
                  f"R²_L={r['r2_l']:.4f}  R²_R={r['r2_r']:.4f}{marker}")

        if best_res["bic_total"] >= res["bic_total"]:
            print("\nNo improvement — stopping.")
            break

        current_terms.append(best_cand)
        res = best_res
        print(f"\n  Added '{term_label(best_cand, display)}'  →  "
              f"BIC={res['bic_total']:.2f}  "
              f"R²_L={res['r2_l']:.4f}  R²_R={res['r2_r']:.4f}")
    else:
        print("\nReached step limit.")

    term_labels = ["Intercept"] + [term_label(t, display) for t in current_terms]
    print(f"\n{'=' * 70}")
    print(f"FINAL — {label}  ({len(current_terms)} terms + intercept)")
    print(f"Terms: {term_labels}")
    print(f"BIC total: {res['bic_total']:.2f}  "
          f"R²_L={res['r2_l']:.4f}  R²_R={res['r2_r']:.4f}")
    print(f"\n{'Coefficient':<26} {'Left':>10}  {'Right':>10}")
    print("-" * 50)
    for lbl, cl, cr in zip(term_labels, res["coef_l"], res["coef_r"]):
        print(f"  {lbl:<24} {cl:>10.6f}  {cr:>10.6f}")

    return {
        "ModelTerms": [list(t) for t in current_terms],
        "L": {lbl: float(round(c, 6))
              for lbl, c in zip(term_labels, res["coef_l"])},
        "R": {lbl: float(round(c, 6))
              for lbl, c in zip(term_labels, res["coef_r"])},
    }

# ── Run both families ─────────────────────────────────────────────────────────
DISPLAY_ETIV  = {"age_open": "age", "sex_enc": "sex", "dm15": "edu", "eTIV": "eTIV"}
DISPLAY_BRAIN = {"age_open": "age", "sex_enc": "sex", "dm15": "edu",
                 "BrainSegVolNotVent": "brain"}

etiv_result  = run_stepwise(
    start_terms   = [("age_open",), ("eTIV",)],
    base_features = ["age_open", "sex_enc", "dm15", "eTIV"],
    display       = DISPLAY_ETIV,
    label         = "eTIV model",
)

brain_result = run_stepwise(
    start_terms   = [("age_open",), ("BrainSegVolNotVent",)],
    base_features = ["age_open", "sex_enc", "dm15", "BrainSegVolNotVent"],
    display       = DISPLAY_BRAIN,
    label         = "Brain model",
)

# ── Write to YAML ─────────────────────────────────────────────────────────────
cfg["Preprocessing"]["Offsets"] = offsets
cfg["Models"] = {
    "eTIV":  etiv_result,
    "Brain": brain_result,
}
# Remove old flat L/R/ModelTerms keys if present from previous runs
for key in ["L", "R", "ModelTerms", "SelectedBaseModel"]:
    cfg.pop(key, None)

with open(CFG_FILE, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"\n\nUpdated {CFG_FILE} with both model families and offsets.")
