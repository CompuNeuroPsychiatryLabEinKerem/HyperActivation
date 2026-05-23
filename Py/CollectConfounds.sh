#!/bin/bash

#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --job-name=CollectConfounds

. /etc/profile.d/huji-lmod.sh

# ── Paths ─────────────────────────────────────────────────────────────────
# Root of the fMRIPrep derivatives tree (source)
BASE='/sci/nosnap/shahar.arzy/meitar2706/Assuta/derivatives/fmriprep'

# Flat destination folder — all confound files land here, no subdirectories
DESTDIR='/sci/labs/shahar.arzy/uri.elias/Projects_Data/Assuta_Meitar/Confounds'

# ── Sanity check ──────────────────────────────────────────────────────────
if [ ! -d "$BASE" ]; then
    echo "ERROR: source directory not found: $BASE"
    exit 1
fi

mkdir -p "$DESTDIR"

# ── Collect ───────────────────────────────────────────────────────────────
# The glob below mirrors the Python:
#   opj('sub-*', '*', 'func', '*confounds_timeseries.tsv')
# i.e.  sub-<subject> / <session> / func / *confounds_timeseries.tsv
#
# $(basename "$src") strips the full path so every file lands flat in DESTDIR,
# exactly as shutil.copy(src, opj(destdir, op.split(src)[-1])) does.

n=0
for src in "$BASE"/sub-*/*/func/*confounds_timeseries.tsv; do
    cp "$src" "$DESTDIR/$(basename "$src")"
    echo "  copied: $(basename "$src")"
    (( n++ ))
done

echo ""
echo "Done — $n file(s) copied to $DESTDIR"
