"""Configuration and constants for the HyperActivation workspace."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_PROJECT_DIR = Path(
    os.environ.get("HYPER_ACTIVATION_PROJECT_DIR", r"C:\Projects\HyperActivation")
).expanduser()
DF_FILENAME = "dfSbjData.pkl"
HIPPO_FILENAME = "HippoVolumes.pkl"


