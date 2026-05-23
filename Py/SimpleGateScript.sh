#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=PyGate

. /etc/profile.d/huji-lmod.sh

# ── Python environment ─────────────────────────────────────────────────────
source /sci/labs/shahar.arzy/uri.elias/PyVirtualNew/bin/activate

# ── Guard ──────────────────────────────────────────────────────────────────
if [ -z "$PY_SCRIPT" ]; then
    echo "ERROR: PY_SCRIPT is not set."
    echo "Usage: sbatch --export=PY_SCRIPT=\"MyScript.py\" SimpleGateScript.sh"
    exit 1
fi
if [ ! -f "$PY_SCRIPT" ]; then
    echo "ERROR: script not found: $PY_SCRIPT"
    exit 1
fi

# ── Info ───────────────────────────────────────────────────────────────────
echo "Host:   $(hostname)"
echo "CPUs:   $SLURM_CPUS_PER_TASK"
echo "Script: $PY_SCRIPT"
echo ""

# ── Run ────────────────────────────────────────────────────────────────────
# PY_ARGS is optional: sbatch --export=PY_SCRIPT="foo.py",PY_ARGS="arg1 arg2" ...
python "$PY_SCRIPT" $PY_ARGS

# Usage: sbatch --export=PY_SCRIPT="Py/AssutaCleanBold.py",PY_ARGS="/path/to/CleanCfg.yaml" Py/SimpleGateScript.sh
