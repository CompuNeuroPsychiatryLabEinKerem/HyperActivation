import yaml
import pandas as pd

# Load model config
with open("HipEst.yml") as f:
    cfg = yaml.safe_load(f)

offsets = cfg["Offsets"]
cols = cfg["Columns"]
coef_l = cfg["L"]
coef_r = cfg["R"]

# Load subject metadata: {id: [Diagnosis, Sex, Age]}
with open(cfg["Files"]["Subjects_Meta"]) as f:
    meta = yaml.safe_load(f)

# Load MRI structural data
mri = pd.read_csv(cfg["Files"]["FS_Stats"])
mri = mri[[cols["subject"], cols["eTIV"], cols["lHip"], cols["rHip"]]]
mri.columns = ["SubjectID", "eTIV", "lHipMeasured", "rHipMeasured"]

# Build metadata dataframe
meta_df = pd.DataFrame.from_dict(
    meta, orient="index", columns=["Diagnosis", "Sex", "Age"]
)
meta_df.index.name = "SubjectID"
meta_df = meta_df.reset_index()

# Merge on SubjectID
df = mri.merge(meta_df, on="SubjectID", how="inner")

missing_mri = set(meta_df["SubjectID"]) - set(mri["SubjectID"])
missing_meta = set(mri["SubjectID"]) - set(meta_df["SubjectID"])
if missing_mri:
    print(f"Warning: subjects in metadata but not in MRI file: {missing_mri}")
if missing_meta:
    print(f"Warning: subjects in MRI file but not in metadata: {missing_meta}")

# Compute offset-adjusted predictors
# global_eTIV from FreeSurfer is in mm³; model offset (1500) is in cm³/mL → divide by 1000
age = df["Age"].astype(float) - offsets["Age"]
etiv = df["eTIV"].astype(float) / 1000.0 - offsets["eTIV"]


def predict(coef, age, etiv):
    result = coef["Intercept"]
    result += coef.get("Age", 0) * age
    result += coef.get("Age2", 0) * age ** 2
    result += coef.get("Age3", 0) * age ** 3
    result += coef.get("eTIV", 0) * etiv
    result += coef.get("eTIV2", 0) * etiv ** 2
    return result


df["lHipPredicted"] = predict(coef_l, age, etiv)
df["rHipPredicted"] = predict(coef_r, age, etiv)

# Round predictions to 2 decimal places
df["lHipPredicted"] = df["lHipPredicted"].round(2)
df["rHipPredicted"] = df["rHipPredicted"].round(2)

output = df[["SubjectID", "Age", "Sex", "Diagnosis",
             "lHipPredicted", "lHipMeasured",
             "rHipPredicted", "rHipMeasured"]]

output.to_csv("hippocampi_estimates.csv", index=False)
print(f"Saved {len(output)} subjects to hippocampi_estimates.csv")
print(output.to_string(index=False))
