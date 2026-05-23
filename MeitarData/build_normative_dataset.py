import pandas as pd
import numpy as np
import re

MRI_FILE  = "../AABC_Data/asegstats.csv"
DEMO_FILE = "../AABC_Data/AABC_Release2_Non-imaging_Data-XL.csv"
OUT_FILE  = "aabc_normative_dataset.csv"

MRI_COLS  = ["Session", "EstimatedTotalIntraCranialVol", "BrainSegVolNotVent",
             "FS_L_Hippocampus", "FS_R_Hippocampus"]
DEMO_COLS = ["id_event", "id", "age_open", "sex", "dm15", "diagnosis_mci_ad"]

# ── Load MRI structural data ─────────────────────────────────────────────────
mri = pd.read_csv(MRI_FILE, usecols=MRI_COLS)

# Parse subject and session number out of Session string
def parse_session(s):
    m = re.match(r"HCA(\d+)_V(\d+)_MR", s)
    return (m.group(1), int(m.group(2))) if m else (None, None)

mri[["SubjectID", "SessionNum"]] = mri["Session"].apply(
    lambda s: pd.Series(parse_session(s))
)
# Key for joining with demo (strip trailing _MR)
mri["id_event"] = mri["Session"].str.replace(r"_MR$", "", regex=True)

# ── Load demographic / non-imaging data ──────────────────────────────────────
# Row 0 = long descriptions (skip), Row 1 = short column names (header)
demo_raw = pd.read_csv(DEMO_FILE, header=1, low_memory=False)

missing = [c for c in DEMO_COLS if c not in demo_raw.columns]
if missing:
    raise ValueError(f"Columns not found in demo file: {missing}")

demo = demo_raw[DEMO_COLS].copy()

# ── Per-session fields: age_open, sex ────────────────────────────────────────
session_demo = demo[["id_event", "age_open", "sex"]].dropna(subset=["id_event"])

# ── Per-subject fields: dm15, diagnosis_mci_ad ───────────────────────────────
# One non-NaN value expected per subject across all rows

def extract_subject_value(col_name):
    """Return dict {id: value} using the single non-NaN per subject; report anomalies."""
    result = {}
    anomalies = []
    for subj_id, grp in demo.groupby("id")[col_name]:
        vals = grp.dropna().unique()
        if len(vals) == 0:
            anomalies.append(f"  [{col_name}] {subj_id}: NO value found")
        elif len(vals) > 1:
            anomalies.append(f"  [{col_name}] {subj_id}: multiple values {list(vals)}")
            result[subj_id] = vals[0]   # take first, flag anyway
        else:
            result[subj_id] = vals[0]
    return result, anomalies

edu_map,   edu_anom   = extract_subject_value("dm15")
diag_map,  diag_anom  = extract_subject_value("diagnosis_mci_ad")

# ── Merge ────────────────────────────────────────────────────────────────────
df = mri.merge(session_demo, on="id_event", how="left")

# subject id in demo uses format HCA{sbj}, derive it from SubjectID
df["id"] = "HCA" + df["SubjectID"]
df["dm15"]              = df["id"].map(edu_map)
df["diagnosis_mci_ad"]  = df["id"].map(diag_map)

# ── Anomaly report ───────────────────────────────────────────────────────────
print("=" * 60)
print("ANOMALY REPORT")
print("=" * 60)

# Sessions in MRI with no match in demo
unmatched_sessions = df[df["age_open"].isna()]["Session"].tolist()
if unmatched_sessions:
    print(f"\nSessions with no age_open in demo ({len(unmatched_sessions)}):")
    for s in unmatched_sessions:
        print(f"  {s}")
else:
    print("\nAll MRI sessions matched to demo age_open: OK")

if edu_anom:
    print(f"\nEducation (dm15) anomalies ({len(edu_anom)}):")
    print("\n".join(edu_anom))
else:
    print("Education (dm15): no anomalies")

if diag_anom:
    print(f"\nDiagnosis anomalies ({len(diag_anom)}):")
    print("\n".join(diag_anom))
else:
    print("Diagnosis: no anomalies")

# ── Save ─────────────────────────────────────────────────────────────────────
out_cols = ["SubjectID", "SessionNum", "Session",
            "age_open", "sex", "dm15", "diagnosis_mci_ad",
            "EstimatedTotalIntraCranialVol", "BrainSegVolNotVent",
            "FS_L_Hippocampus", "FS_R_Hippocampus"]

df[out_cols].to_csv(OUT_FILE, index=False)

print(f"\n{'=' * 60}")
print(f"Saved {len(df)} rows to {OUT_FILE}")
print(f"Subjects: {df['SubjectID'].nunique()}")
print(f"\nDiagnosis counts (1=Unimpaired, 2=MCI, 3=AD):")
print(df.drop_duplicates('SubjectID')['diagnosis_mci_ad'].value_counts().sort_index().to_string())
print(f"\nSample:")
print(df[out_cols].head(6).to_string(index=False))
