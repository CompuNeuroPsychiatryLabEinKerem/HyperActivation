"""Data loading utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from . import settings


class DataRepository:
    """Handles loading and caching of project resources."""

    def __init__(self, project_dir: Path | str | None = None) -> None:
        self.project_dir = Path(project_dir or settings.DEFAULT_PROJECT_DIR)
        self._subject_df: pd.DataFrame | None = None
        self._hippo_volumes: Dict | None = None

    @property
    def subject_df(self) -> pd.DataFrame:
        if self._subject_df is None:
            self._subject_df = self._read_pickle(settings.DF_FILENAME)
        return self._subject_df

    @property
    def hippo_volumes(self) -> Dict:
        if self._hippo_volumes is None:
            self._hippo_volumes = self._read_pickle(settings.HIPPO_FILENAME)
        return self._hippo_volumes

    def get_connectivity_terms(self) -> List[str]:
        """Return available keyword terms for bilateral connectivity columns."""
        df = self.subject_df
        parts: Iterable[List[str]] = (c.split("_")[3:5] for c in df.columns if c.startswith("RSP_TO_Bilaterals"))
        terms = sorted({term for pair in parts for term in pair if term})
        return terms

    def _read_pickle(self, filename: str):
        path = self.project_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected pickle at {path}")
        with open(path, "rb") as fh:
            return pickle.load(fh)


