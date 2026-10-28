"""Sample NLCD canopy and impervious rasters at lon/lat points."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer


class RasterSampler:
    """Keeps a GeoTIFF open and returns values at lon/lat. Out-of-range or nodata -> NaN."""

    def __init__(self, path: Path, valid_max: float = 100.0):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Raster not found: {self.path}. See README > Data.")
        self.src = rasterio.open(self.path)
        self.to_crs = Transformer.from_crs("EPSG:4326", self.src.crs, always_xy=True)
        self.valid_max = valid_max

    def sample(self, lon, lat) -> np.ndarray:
        lon = np.atleast_1d(np.asarray(lon, dtype="float64"))
        lat = np.atleast_1d(np.asarray(lat, dtype="float64"))
        xs, ys = self.to_crs.transform(lon, lat)
        out = np.full(lon.shape, np.nan)
        b = self.src.bounds
        inside = (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top)
        if inside.any():
            vals = np.array([v[0] for v in self.src.sample(zip(xs[inside], ys[inside]))], dtype="float64")
            nod = self.src.nodata
            bad = (vals > self.valid_max) | (vals < 0)
            if nod is not None:
                bad |= vals == nod
            vals[bad] = np.nan  # NLCD uses 254/255 or 127/250 as fill values
            out[inside] = vals
        return out

    def close(self):
        self.src.close()


class LandCover:
    """Samples every land-cover raster at once. Optional rasters are simply skipped
    if they have not been built yet, so the basic pipeline still runs."""

    def __init__(self, canopy_tif: Path, impervious_tif: Path, extra: dict = None):
        self.samplers = {
            "canopy_pct": RasterSampler(canopy_tif),
            "impervious_pct": RasterSampler(impervious_tif),
        }
        for name, (path, vmax) in (extra or {}).items():
            if Path(path).exists():
                self.samplers[name] = RasterSampler(path, valid_max=vmax)

    @classmethod
    def from_config(cls, cfg) -> "LandCover":
        # Derived rasters are optional: a config that predates scripts.build_features
        # (or a test fixture) still gets the two base layers.
        optional = {
            "canopy_100m": ("canopy_nbr_tif", 100.0),
            "impervious_100m": ("impervious_nbr_tif", 100.0),
            "dist_coast_km": ("dist_coast_tif", 1e4),   # LA box spans ~0-120 km from shore
            "elevation_m": ("elevation_tif", 9000.0),   # Mt Baldy tops out near 3000 m
        }
        extra = {name: (cfg.path("data", key), vmax)
                 for name, (key, vmax) in optional.items() if key in cfg["data"]}
        return cls(cfg.path("data", "canopy_tif"), cfg.path("data", "impervious_tif"), extra)

    def features(self, lon, lat) -> dict[str, np.ndarray]:
        return {name: s.sample(lon, lat) for name, s in self.samplers.items()}

    def close(self):
        for s in self.samplers.values():
            s.close()
