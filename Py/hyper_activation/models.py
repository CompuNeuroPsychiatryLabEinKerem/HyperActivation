"""Dataclasses for UI state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple


@dataclass(frozen=True)
class SelectionState:
    domain: str = "Space"
    rsp_vertices: str = "Full"
    rsp_source: str = "Default"
    hpc_size: str = "200"
    conn_units: str = "Singles"
    conn_stat: str = "mean"
    conn_threshold: float = 0.3
    conn_keywords: Tuple[str, ...] = ()
    hp_source: str = "FS"
    hp_stat: str = "mean"
    penalty: str = "none"
    transformation: str = "score"  # "score", "log(score)", "1/score"
    regress_age: str = "none"
    color_margin: float = 1.3
    c_axis_visible: bool = True
    num_permutations: int = 1000
    use_loo: bool = True
    reference_value: str = "median"
    enable_classification: bool = True
    remove_rm: bool = False
    remove_ad: bool = False

    @classmethod
    def from_controls(cls, **kwargs) -> "SelectionState":
        keywords: Sequence[str] = kwargs.pop("conn_keywords", ())
        return cls(conn_keywords=tuple(keywords), **kwargs)

