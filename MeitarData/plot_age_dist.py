import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("aabc_normative_dataset.csv")

# One row per subject (first session) to avoid counting repeat visits
df["age_open"] = pd.to_numeric(df["age_open"], errors="coerce")
subj = df.sort_values("SessionNum").groupby("SubjectID").first().reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Age distribution by sex  (one entry per subject, baseline session)", fontsize=13)

colors = {"M": "#2196F3", "F": "#E91E63"}
bins = np.arange(subj["age_open"].min() - 1, subj["age_open"].max() + 3, 2)

# ── Left: overlapping histograms ─────────────────────────────────────────────
ax = axes[0]
for sex, grp in subj.groupby("sex"):
    ax.hist(grp["age_open"], bins=bins, alpha=0.55,
            color=colors.get(sex, "gray"), label=sex, edgecolor="white", linewidth=0.4)
ax.set_xlabel("Age (years)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Histogram")
ax.legend(title="Sex")
ax.grid(True, alpha=0.25)

# ── Right: KDE + rug ─────────────────────────────────────────────────────────
from scipy.stats import gaussian_kde

ax = axes[1]
x = np.linspace(subj["age_open"].min() - 2, subj["age_open"].max() + 2, 300)
for sex, grp in subj.groupby("sex"):
    ages = grp["age_open"].dropna().values
    kde  = gaussian_kde(ages, bw_method=0.3)
    ax.plot(x, kde(x), color=colors.get(sex, "gray"), lw=2, label=sex)
    ax.fill_between(x, kde(x), alpha=0.15, color=colors.get(sex, "gray"))
    ax.plot(ages, np.full_like(ages, -0.0005), "|",
            color=colors.get(sex, "gray"), alpha=0.5, markersize=8)

ax.set_xlabel("Age (years)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("KDE")
ax.legend(title="Sex")
ax.grid(True, alpha=0.25)

# ── Counts in legend ─────────────────────────────────────────────────────────
counts = subj["sex"].value_counts()
for ax in axes:
    handles, labels = ax.get_legend_handles_labels()
    labels = [f"{l}  (n={counts.get(l, 0)})" for l in labels]
    ax.legend(handles, labels, title="Sex")

fig.tight_layout()
plt.show()
