"""Candidate walking routes from openrouteservice, plus point sampling along a route."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np
import requests

from .config import Config

EARTH_R = 6_371_000.0


def _post_with_backoff(http, url, body, headers, timeout, max_retries=4):
    """POST that waits out a per-minute 429 instead of treating it as a hard stop.

    A 429 is "try again shortly" (freeroute even tells us how long via `retryAfter`),
    unlike a 403 quota-exhausted error, which genuinely means stop for the day. Retrying
    here keeps that distinction out of the caller, which only ever sees a real failure.
    """
    delay = 5.0
    for attempt in range(max_retries + 1):
        r = http.post(url, json=body, headers=headers, timeout=timeout)
        if r.status_code != 429 or attempt == max_retries:
            return r
        wait = delay
        try:
            wait = max(wait, float(r.json().get("retryAfter", delay)))
        except Exception:
            pass
        time.sleep(wait + 1)
        delay *= 2
    return r


@dataclass
class Route:
    name: str
    coords: list[tuple[float, float]]  # (lon, lat)
    distance_m: float
    source: str  # "ors" or "ors-via"


def haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def densify(coords: list[tuple[float, float]], spacing_m: float) -> np.ndarray:
    """Points every `spacing_m` metres along the polyline (linear interpolation per segment)."""
    pts = [coords[0]]
    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
        d = float(haversine(x1, y1, x2, y2))
        n = max(1, int(math.ceil(d / spacing_m)))
        for k in range(1, n + 1):
            t = k / n
            pts.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    return np.asarray(pts)


def _overlap(a: Route, b: Route, tol_deg: float = 2e-4) -> float:
    """Share of a's sampled points that lie within ~20 m of b."""
    pa, pb = densify(a.coords, 50), densify(b.coords, 50)
    d = np.abs(pa[:, None, :] - pb[None, :, :]).max(axis=2).min(axis=1)
    return float((d < tol_deg).mean())


def _candidates_via(directions_fn, cfg, start, end, source_tag: str) -> list[Route]:
    """Shared by every directions provider: native alternatives plus side-point detours,
    deduplicated by path overlap. `directions_fn(coords, alternatives) -> list[geojson feature]`
    is the only thing that differs between providers.
    """
    routes: list[Route] = []

    def add(feat: dict, source: str):
        coords = [tuple(c) for c in feat["geometry"]["coordinates"]]
        r = Route("", coords, float(feat["properties"]["summary"]["distance"]), source)
        if all(_overlap(r, q) < 0.9 for q in routes):
            routes.append(r)

    for f in directions_fn([start, end], alternatives=True):
        add(f, source_tag)

    (x1, y1), (x2, y2) = start, end
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    for off in cfg.get("via_offsets", []):
        via = (mx - dy * off, my + dx * off)  # perpendicular side point
        try:
            for f in directions_fn([start, via, end], alternatives=False)[:1]:
                add(f, f"{source_tag}-via")
        except RuntimeError:
            continue  # via point may land somewhere unroutable; skip it

    for i, r in enumerate(routes, 1):
        r.name = f"R{i}"
    if not routes:
        raise RuntimeError("Routing provider returned no route between these points.")
    return routes


class ORSClient:
    def __init__(self, cfg: Config, api_key: str | None = None, session: requests.Session | None = None):
        self.cfg = cfg["routing"]
        self.key = api_key or os.environ.get("ORS_API_KEY")
        if not self.key:
            raise RuntimeError("Set ORS_API_KEY (free key from openrouteservice.org).")
        self.http = session or requests.Session()

    def _directions(self, coords: list[tuple[float, float]], alternatives: bool) -> list[dict]:
        body: dict = {"coordinates": [list(c) for c in coords]}
        if alternatives and len(coords) == 2:
            body["alternative_routes"] = {
                "target_count": self.cfg["alternatives"],
                "weight_factor": self.cfg["weight_factor"],
                "share_factor": self.cfg["share_factor"],
            }
        url = f"{self.cfg['base_url']}/v2/directions/{self.cfg['profile']}/geojson"
        r = _post_with_backoff(
            self.http, url, body,
            {"Authorization": self.key, "Content-Type": "application/json"},
            self.cfg["timeout_s"],
        )
        if r.status_code != 200:
            raise RuntimeError(f"openrouteservice error {r.status_code}: {r.text[:300]}")
        return r.json().get("features", [])

    def candidates(self, start: tuple[float, float], end: tuple[float, float]) -> list[Route]:
        """ORS alternatives, plus routes through side points for more variety on short trips."""
        return _candidates_via(self._directions, self.cfg, start, end, "ors")

    def suggest(self, text: str, cfg_bbox, size: int = 6) -> list[tuple[str, float, float]]:
        """Place name -> up to `size` (label, lon, lat) matches inside the LA box.

        /geocode/autocomplete often returns nothing for partial venue names
        ("troy east"), so this uses /geocode/search, which still fuzzy-matches.
        """
        r = self.http.get(
            f"{self.cfg['base_url']}/geocode/search",
            params={
                "api_key": self.key,
                "text": text,
                "size": size,
                "boundary.rect.min_lon": cfg_bbox.min_lon,
                "boundary.rect.min_lat": cfg_bbox.min_lat,
                "boundary.rect.max_lon": cfg_bbox.max_lon,
                "boundary.rect.max_lat": cfg_bbox.max_lat,
            },
            timeout=self.cfg["timeout_s"],
        )
        r.raise_for_status()
        out = []
        for f in r.json().get("features", []):
            lon, lat = f["geometry"]["coordinates"]
            out.append((f["properties"].get("label", text), float(lon), float(lat)))
        return out

    def geocode(self, text: str, cfg_bbox) -> tuple[float, float]:
        """Place name -> (lon, lat) of the best match, restricted to the LA box."""
        hits = self.suggest(text, cfg_bbox, size=1)
        if not hits:
            raise ValueError(f"No place found for '{text}' inside the LA area.")
        return hits[0][1], hits[0][2]


class FreeRouteClient:
    """freeroute.org, a second free directions API (independent daily quota from ORS).

    Its /v1/directions/{profile} endpoint turned out to accept the exact same request
    body shape as ORS (coordinates list, alternative_routes block) and return the same
    GeoJSON FeatureCollection shape, just without the "/geojson" suffix in the path and
    with the key sent as a Bearer token instead of a raw header value. No public docs
    were reachable (the site is a client-rendered app); this was confirmed empirically
    against the live API before being wired in. No geocoding endpoint is used here --
    the app's existing ORS-backed geocoder is unaffected.
    """

    def __init__(self, cfg: Config, api_key: str | None = None, session: requests.Session | None = None):
        self.cfg = cfg["routing"]
        self.key = api_key or os.environ.get("FREEROUTE_API_KEY")
        if not self.key:
            raise RuntimeError("Set FREEROUTE_API_KEY (from freeroute.org).")
        self.http = session or requests.Session()
        self.base_url = cfg.get("freeroute", {}).get("base_url", "https://api.maps.freeroute.org")

    def _directions(self, coords: list[tuple[float, float]], alternatives: bool) -> list[dict]:
        body: dict = {"coordinates": [list(c) for c in coords]}
        if alternatives and len(coords) == 2:
            body["alternative_routes"] = {
                "target_count": self.cfg["alternatives"],
                "weight_factor": self.cfg["weight_factor"],
                "share_factor": self.cfg["share_factor"],
            }
        url = f"{self.base_url}/v1/directions/{self.cfg['profile']}"
        r = _post_with_backoff(
            self.http, url, body,
            # the key must never reach a query string: 404 pages echo the full URL back,
            # which would leak it into logs/error messages
            {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            self.cfg["timeout_s"],
        )
        if r.status_code != 200:
            raise RuntimeError(f"freeroute error {r.status_code}: {r.text[:300]}")
        return r.json().get("features", [])

    def candidates(self, start: tuple[float, float], end: tuple[float, float]) -> list[Route]:
        return _candidates_via(self._directions, self.cfg, start, end, "freeroute")
