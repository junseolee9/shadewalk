"""Detour filter + weighted score + minimum-cooling rule."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RouteStats:
    name: str
    distance_m: float
    mean_lst_c: float
    peak_lst_c: float
    shade_pct: float
    eligible: bool = True
    score: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Decision:
    chosen: RouteStats
    shortest: RouteStats
    ranked: list[RouteStats]
    reason: str


def _norm(v: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(v), np.nanmax(v)
    return np.zeros_like(v) if hi - lo < 1e-9 else (v - lo) / (hi - lo)


def decide(stats: list[RouteStats], weights: dict, max_detour: float, min_cooling_c: float) -> Decision:
    shortest = min(stats, key=lambda s: s.distance_m)
    for s in stats:
        s.eligible = s.distance_m <= max_detour * shortest.distance_m + 1e-6
    el = [s for s in stats if s.eligible]

    heat = _norm(np.array([s.mean_lst_c for s in el]))
    shade = _norm(np.array([s.shade_pct for s in el]))
    dist = _norm(np.array([s.distance_m for s in el]))
    for s, h, sh, d in zip(el, heat, shade, dist):
        s.score = float(weights["heat"] * h + weights["no_shade"] * (1 - sh) + weights["distance"] * d)

    best = min(el, key=lambda s: s.score)
    ranked = sorted(stats, key=lambda s: (s.score is None, s.score if s.score is not None else 0))

    if best is shortest:
        reason = "The shortest route is already the best-scoring eligible route."
    elif shortest.mean_lst_c - best.mean_lst_c < min_cooling_c:
        gain = shortest.mean_lst_c - best.mean_lst_c
        reason = (f"{best.name} is only {gain:.1f} °C cooler (< {min_cooling_c} °C), so the shortest route is kept.")
        best = shortest
    else:
        reason = (
            f"{best.name} is {shortest.mean_lst_c - best.mean_lst_c:.1f} °C cooler for "
            f"+{best.distance_m - shortest.distance_m:.0f} m."
        )
    return Decision(best, shortest, ranked, reason)
