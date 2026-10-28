"""Read ECOSTRESS ECO_L2T_LSTE v002 granules (one GeoTIFF per layer).

Expected file names, as delivered by Earthdata Search:
    ECOv002_L2T_LSTE_<orbit>_<scene>_<tile>_<YYYYMMDDTHHMMSS>_<build>_<ver>_<layer>.tif
Layers used: LST, QC, cloud, water. The timestamp is UTC.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

from .config import BBox

NAME_RE = re.compile(
    r"^(?P<key>ECOv002_L2T_LSTE_\d+_\d+_(?P<tile>\w+?)_(?P<ts>\d{8}T\d{6})_\d+_\d+)_(?P<layer>[A-Za-z_]+)\.tif$"
)


@dataclass
class Granule:
    key: str
    tile: str
    time_utc: datetime
    layers: dict[str, Path] = field(default_factory=dict)

    def local_time(self, tz: str) -> datetime:
        return self.time_utc.astimezone(ZoneInfo(tz))


def find_granules(folder: Path) -> list[Granule]:
    """Group layer files by granule. Granules without an LST layer are skipped."""
    granules: dict[str, Granule] = {}
    for p in sorted(Path(folder).rglob("*.tif")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        g = granules.get(m["key"])
        if g is None:
            t = datetime.strptime(m["ts"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            g = granules[m["key"]] = Granule(m["key"], m["tile"], t)
        g.layers[m["layer"]] = p
    return [g for g in granules.values() if "LST" in g.layers]


def to_celsius(arr: np.ndarray, src: rasterio.DatasetReader) -> np.ndarray:
    """Return LST in deg C with invalid pixels as NaN.

    v002 tiles normally store float32 Kelvin. If a file is integer-scaled, apply the
    scale/offset stored in the GeoTIFF tags (ECOSTRESS uses 0.02 K for scaled LST).
    """
    out = arr.astype("float64")
    if src.nodata is not None:
        out[arr == src.nodata] = np.nan
    if np.issubdtype(arr.dtype, np.integer):
        scale = (src.scales or (1.0,))[0]
        offset = (src.offsets or (0.0,))[0]
        if scale == 1.0 and offset == 0.0:
            scale = 0.02  # documented ECOSTRESS LST scale factor
        out = out * scale + offset
        out[arr == 0] = np.nan
    out[out <= 0] = np.nan
    return out - 273.15


def _read_mask(path: Path | None, shape) -> np.ndarray | None:
    if path is None:
        return None
    with rasterio.open(path) as s:
        a = s.read(1)
    return a if a.shape == shape else None


def granule_pixels(
    g: Granule,
    bbox: BBox,
    tz: str,
    keep_qc: list[int],
    valid_range_c: tuple[float, float],
    n_samples: int,
    rng: np.random.Generator,
    max_lst_err_k: float = float("inf"),
    robust_z: float = float("inf"),
) -> tuple[pd.DataFrame, dict]:
    """Return a random sample of valid pixels plus per-stage pixel counts.

    The counts feed the data funnel table in the report, so every filter records how
    many pixels it removed rather than silently dropping them.
    """
    with rasterio.open(g.layers["LST"]) as src:
        raw = src.read(1)
        lst = to_celsius(raw, src)
        transform, crs = src.transform, src.crs

    # lon/lat of every pixel centre, so "inside the box" can be measured before masking
    rr, cc = np.indices(lst.shape)
    xs, ys = rasterio.transform.xy(transform, rr.ravel(), cc.ravel(), offset="center")
    lon_all, lat_all = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        np.asarray(xs), np.asarray(ys)
    )
    in_box = np.asarray(bbox.contains(lon_all, lat_all)).reshape(lst.shape)

    stats = {"granule": g.key, "px_in_box": int(in_box.sum())}
    valid = in_box & np.isfinite(lst) & (lst >= valid_range_c[0]) & (lst <= valid_range_c[1])
    stats["px_lst_range"] = int(valid.sum())

    cloud = _read_mask(g.layers.get("cloud"), lst.shape)
    if cloud is not None:
        valid &= cloud == 0
    stats["px_cloud"] = int(valid.sum())

    water = _read_mask(g.layers.get("water"), lst.shape)
    if water is not None:
        valid &= water == 0
    stats["px_water"] = int(valid.sum())

    qc = _read_mask(g.layers.get("QC"), lst.shape)
    if qc is not None:
        valid &= np.isin(qc.astype("uint16") & 0b11, keep_qc)
    stats["px_qc"] = int(valid.sum())

    # LST_err is the per-pixel uncertainty in K; large values flag thin cloud / bad retrieval
    err = _read_mask(g.layers.get("LST_err"), lst.shape)
    if err is not None and np.isfinite(max_lst_err_k):
        with np.errstate(invalid="ignore"):
            valid &= np.isfinite(err) & (err <= max_lst_err_k)
    stats["px_lst_err"] = int(valid.sum())

    # Robust within-scene outlier cut: cloud shadow survives the masks above but sits
    # far below the scene median. MAD is used instead of std so the cut is not dragged
    # by the very outliers it is meant to remove.
    vals = lst[valid]
    if vals.size and np.isfinite(robust_z):
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        if mad > 0:
            keep = np.abs(lst - med) <= robust_z * 1.4826 * mad  # 1.4826: MAD -> sigma
            valid &= np.isfinite(lst) & keep
    stats["px_robust"] = int(valid.sum())
    stats["valid_fraction"] = stats["px_robust"] / stats["px_in_box"] if stats["px_in_box"] else 0.0

    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return pd.DataFrame(), stats

    flat = rows * lst.shape[1] + cols
    lon, lat, vals = lon_all[flat], lat_all[flat], lst[rows, cols]
    height = _read_mask(g.layers.get("height"), lst.shape)  # ECOSTRESS ships a DEM layer

    if vals.size > n_samples:
        pick = rng.choice(vals.size, n_samples, replace=False)
        rows, cols, lon, lat, vals = rows[pick], cols[pick], lon[pick], lat[pick], vals[pick]
    stats["px_sampled"] = int(vals.size)

    local = g.local_time(tz)
    out = pd.DataFrame(
        {
            "lon": lon,
            "lat": lat,
            "lst_c": vals,
            "hour": local.hour + local.minute / 60.0,
            "date": local.date().isoformat(),
            "granule": g.key,
        }
    )
    if height is not None:
        out["elevation_m"] = height[rows, cols].astype("float64")
    return out, stats
