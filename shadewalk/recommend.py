"""Recommend a walking route for *right now*, using the system clock in LA time."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config
from .dataset import features_for, is_daytime
from .landcover import LandCover
from .routing import ORSClient, Route, densify
from .scoring import Decision, RouteStats, decide


@dataclass
class Recommendation:
    local_time: datetime
    daytime: bool
    decision: Decision
    routes: dict[str, Route]
    target: str = "lst"

    def summary(self) -> str:
        d = self.decision
        head = "mean dLST" if self.target == "dlst" else "mean LST"
        lines = [f"Local time: {self.local_time:%Y-%m-%d %H:%M %Z}"]
        if not self.daytime:
            lines.append("Outside the daytime window: heat model not used, showing the shortest route.")
        lines.append(f"Recommended: {d.chosen.name} ({d.chosen.distance_m:.0f} m). {d.reason}")
        lines.append(f"{'route':<6}{'length':>9}{head:>10}{'peak':>8}{'shade':>8}{'score':>7}  status")
        for s in d.ranked:
            status = "chosen" if s is d.chosen else ("shortest" if s is d.shortest else ("ok" if s.eligible else "too long"))
            score = "-" if s.score is None else f"{s.score:.2f}"
            mean = "-" if np.isnan(s.mean_lst_c) else f"{s.mean_lst_c:.1f}C"
            peak = "-" if np.isnan(s.peak_lst_c) else f"{s.peak_lst_c:.1f}C"
            lines.append(
                f"{s.name:<6}{s.distance_m:>8.0f}m{mean:>10}{peak:>8}{s.shade_pct:>7.1f}%{score:>7}  {status}"
            )
        return "\n".join(lines)


class Recommender:
    def __init__(self, cfg: Config, model, router: ORSClient, landcover: LandCover,
                 features=None, target="lst"):
        self.cfg, self.router, self.lc = cfg, router, landcover
        # model.load returns (estimator, features, target); accept that tuple directly
        if isinstance(model, tuple):
            model, features, target = model
        self.model = model
        self.features = list(features) if features else features_for(cfg)
        # "dlst" predicts the within-scene anomaly. Routes are compared inside one
        # scene, so the anomaly orders them exactly as absolute temperature would.
        self.target = target

    def _stats(self, route: Route, hour: float, use_model: bool) -> RouteStats:
        pts = densify(route.coords, self.cfg["routing"]["sample_spacing_m"])
        lc = self.lc.features(pts[:, 0], pts[:, 1])
        canopy = np.nan_to_num(lc["canopy_pct"], nan=0.0)
        mean_t = peak_t = float("nan")
        if use_model:
            cols = {"lat": pts[:, 1], "lon": pts[:, 0], "hour": hour}
            for name, vals in lc.items():
                # a route can clip a nodata pixel; fall back to the route's own mean
                fill = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
                cols[name] = np.nan_to_num(vals, nan=fill)
            X = pd.DataFrame(cols)[self.features]
            t = self.model.predict(X)
            mean_t, peak_t = float(t.mean()), float(t.max())
        return RouteStats(route.name, route.distance_m, mean_t, peak_t, float(canopy.mean()))

    def recommend(self, start: tuple[float, float], end: tuple[float, float], now: datetime | None = None) -> Recommendation:
        tz = ZoneInfo(self.cfg["timezone"])
        local = (now or datetime.now(tz)).astimezone(tz)
        hour = local.hour + local.minute / 60.0
        day = is_daytime(hour, self.cfg)

        routes = self.router.candidates(start, end)
        stats = [self._stats(r, hour, use_model=day) for r in routes]

        if day:
            sc = self.cfg["scoring"]
            decision = decide(stats, sc["weights"], sc["max_detour"], sc["min_cooling_c"])
        else:
            shortest = min(stats, key=lambda s: s.distance_m)
            decision = Decision(shortest, shortest, sorted(stats, key=lambda s: s.distance_m),
                                "Night-time: safety/lighting scoring is future work.")
        return Recommendation(local, day, decision, {r.name: r for r in routes}, self.target)
