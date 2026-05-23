"""Core measurement routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_access import DataRepository
from .models import SelectionState

AXES_NAMES = [
    "RSP_Activation",
    "Connectivity",
    "BlockConnectivity",
    "HippocampiVolume",
    "OrientationPerformance",
    "Age",
    "HPC_Activation",
]


def _regress_age(score: np.ndarray, age: np.ndarray, fix_to_mean_age: bool = True) -> np.ndarray:
    slope, intercept = np.polyfit(age, score, 1)
    residuals = score - (slope * age + intercept)
    if fix_to_mean_age:
        residuals += slope * age.mean() + intercept
    return residuals


def _measure_activation(df: pd.DataFrame, domain: str, num_vertices: int, source: str = "Default") -> pd.Series:
    if num_vertices < 0:
        num_vertices = 0
    measure_name = f"Max_{num_vertices}" if num_vertices > 0 else "Sum"
    # Updated column name pattern: domain_source_Rsp_stat (e.g., Space_PSC_Rsp_Sum)
    # Only for Space domain currently
    if domain == "Space":
        column_name = f"{domain}_{source}_Rsp_{measure_name}"
    else:
        # For other domains, use old pattern (domain_Rsp_stat) if source is Default
        if source.lower() == "default":
            column_name = f"{domain}_Rsp_{measure_name}"
        else:
            # Try new pattern for non-Space domains too
            column_name = f"{domain}_{source}_Rsp_{measure_name}"
    
    if column_name not in df.columns:
        # Fallback to old pattern if new column doesn't exist
        column_name = f"{domain}_Rsp_{measure_name}"
    
    return df[column_name]


def _measure_age(df: pd.DataFrame) -> pd.Series:
    """Return age column from dataframe."""
    return df["DMG_Age"]


def _measure_hpc_activation(df: pd.DataFrame, domain: str, size: str) -> pd.Series:
    """
    Measure HPC activation from dataframe.
    
    Args:
        df: DataFrame with subject data
        domain: Domain name (e.g., 'Space')
        size: Size value ('200', '100', '50')
    
    Returns:
        Series with HPC activation values
    """
    column_name = f"{domain}_Z_Hpc_Max_{size}"
    if column_name not in df.columns:
        raise ValueError(f"Column {column_name} not found in dataframe")
    return df[column_name]


def _measure_connectivity(df: pd.DataFrame, *, units: str, stat: str, thr: float, keywords: tuple[str, ...]) -> pd.Series:
    pref = f"RSP_TO_{units}"
    columns = [c for c in df.columns if c.startswith(pref)]

    # Filter columns based on keywords if provided
    if keywords:
        # For Bilaterals, columns are like RSP_TO_Bilaterals_TERM1_TERM2
        # Extract terms from column names and filter
        filtered_columns = []
        for col in columns:
            # Split column name and extract terms (positions 3 and 4 after RSP_TO_Bilaterals)
            parts = col.split("_")
            if len(parts) >= 5:
                # Extract the two terms (e.g., from "RSP_TO_Bilaterals_TERM1_TERM2")
                col_terms = set(parts[3:5])  # Terms at positions 3 and 4
                # Check if any selected keyword matches any term in this column
                if any(keyword in col_terms for keyword in keywords):
                    filtered_columns.append(col)
            else:
                # For other units or column formats, check if keyword appears in column name
                if any(keyword in col for keyword in keywords):
                    filtered_columns.append(col)
        
        if filtered_columns:
            columns = filtered_columns

    if len(columns) == 1:
        return df[columns[0]]

    vals = df[columns].values
    if stat == "mean":
        measure = vals.mean(1)
    elif stat == "count":
        measure = (vals > thr).mean(1)
    elif stat == "z":
        measure = vals.mean(1) / vals.std(1)
    else:
        raise ValueError(f"Unsupported connectivity stat '{stat}'")

    return pd.Series(measure, index=df.index)


def _measure_performance(df: pd.DataFrame, *, domain: str, penalty: str, transformation: str, regress_age_option: str) -> pd.Series:
    print('Score calculation!!', penalty)
    
    scores = df[f"EC_{domain}"].values
    age = df.DMG_Age.values

    if regress_age_option == "before":
        scores = _regress_age(scores, age)

    if penalty in ["full", "sqrt", "fullinv", "sqrtinv"]:
        f = (df.EC_Space_D2 / df.EC_Space_D1).values
        if penalty.startswith("sqrt"):
            f = np.sqrt(f)
        if penalty.endswith("inv"):
            f = 1 / f
        scores *= f

    # Apply transformation: "score", "log(score)", "1/score"
    if transformation == "log(score)":
        z = np.log(scores / np.median(scores))
    elif transformation == "1/score":
        z = 1.0 / scores
    else:  # "score"
        z = scores

    if regress_age_option == "after":
        z = _regress_age(z, age)

    return pd.Series(z, index=df.index)


def _measure_hippo_volume(volumes, *, source: str, stat: str) -> pd.Series:
    data = volumes[source]
    stat_funcs = dict(
        mean=np.mean,
        left=lambda vs: vs[0],
        right=lambda vs: vs[1],
        min=np.min,
        max=np.max,
    )
    func = stat_funcs[stat]
    subjects, values = zip(*((sbj, func(vols)) for sbj, vols in data.items()))
    return pd.Series(values, index=subjects)


def _measure_task_connectivity(**kwargs):
    # Placeholder: task connectivity metric has not been defined yet.
    raise NotImplementedError("Task connectivity measurement is pending specification.")


MODALITY_TO_FUNC = {
    "RSP_Activation": _measure_activation,
    "Connectivity": _measure_connectivity,
    "BlockConnectivity": _measure_task_connectivity,
    "HippocampiVolume": _measure_hippo_volume,
    "OrientationPerformance": _measure_performance,
    "Age": _measure_age,
    "HPC_Activation": _measure_hpc_activation,
}


class MeasurementService:
    """Maps UI selections to concrete pandas Series."""

    def __init__(self, repository: DataRepository):
        self.repo = repository

    def series_for_modality(self, modality: str, selections: SelectionState) -> pd.Series:
        df = self.repo.subject_df.copy()
        
        # Apply filtering based on selection flags
        if selections.remove_rm:
            # Remove RM: keep only rows where Dg2 != 'RM'
            df = df[df.Dg2 != 'RM']
        if selections.remove_ad:
            # Remove AD: keep only rows where Dg1 != 'AD'
            df = df[df.Dg1 != 'AD']
        
        hippo = self.repo.hippo_volumes

        if modality == "RSP_Activation":
            vertices = selections.rsp_vertices
            num_vertices = int(vertices) if vertices.isdigit() else -1
            return _measure_activation(df, selections.domain, num_vertices, source=selections.rsp_source)

        if modality == "Connectivity":
            return _measure_connectivity(
                df,
                units=selections.conn_units,
                stat=selections.conn_stat,
                thr=selections.conn_threshold,
                keywords=selections.conn_keywords,
            )

        if modality == "BlockConnectivity":
            return _measure_task_connectivity()

        if modality == "HippocampiVolume":
            return _measure_hippo_volume(hippo, source=selections.hp_source, stat=selections.hp_stat)

        if modality == "OrientationPerformance":
            return _measure_performance(
                df,
                domain=selections.domain,
                penalty=selections.penalty,
                transformation=selections.transformation,
                regress_age_option=selections.regress_age,
            )

        if modality == "Age":
            return _measure_age(df)

        if modality == "HPC_Activation":
            return _measure_hpc_activation(df, selections.domain, selections.hpc_size)

        raise KeyError(f"Unknown modality '{modality}'")

