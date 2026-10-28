"""Derived rasters: neighbourhood land cover and distance to the coast.

A pixel's own 30 m canopy value says little about how shaded a *walk* is - what
matters is the tree cover of the surrounding block. These helpers pre-compute the
smoothed rasters once so the per-point sampling stays a cheap lookup.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage

log = logging.getLogger(__name__)


def _write_like(src: rasterio.DatasetReader, data: np.ndarray, path, nodata=np.nan):
    meta = src.meta.copy()
    meta.update(dtype="float32", count=1, nodata=nodata, compress="deflate", tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **meta) as out:
        out.write(data.astype("float32"), 1)


def neighbourhood_mean(src_path, dst_path, radius_m: float = 100.0) -> int:
    """Mean of a percent raster over a square window of about `radius_m`.

    nodata pixels are excluded from both the sum and the count, so coastal and
    edge pixels average only over real land instead of being dragged toward zero.
    """
    with rasterio.open(src_path) as ds:
        a = ds.read(1).astype("float32")
        res = ds.res[0]
        valid = np.ones(a.shape, bool)
        if ds.nodata is not None:
            valid &= a != ds.nodata
        valid &= (a >= 0) & (a <= 100)

        size = max(3, int(round(2 * radius_m / res)) | 1)  # odd window covering the radius
        vals = np.where(valid, a, 0.0)
        num = ndimage.uniform_filter(vals, size=size, mode="nearest")
        den = ndimage.uniform_filter(valid.astype("float32"), size=size, mode="nearest")
        out = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
        _write_like(ds, out, dst_path)
    return size


def distance_to_coast(src_path, dst_path) -> float:
    """Kilometres to the Pacific, taken from the NLCD footprint.

    NLCD only covers land, so its nodata region west of the city *is* the ocean.
    Using the largest connected nodata blob avoids treating small inland holes as sea,
    which means no extra coastline download is needed.
    """
    with rasterio.open(src_path) as ds:
        a = ds.read(1)
        res = ds.res[0]
        nodata = a == ds.nodata if ds.nodata is not None else np.zeros(a.shape, bool)
        if not nodata.any():
            raise ValueError(f"{src_path} has no nodata pixels; cannot locate the coast.")
        lab, n = ndimage.label(nodata)
        sizes = ndimage.sum(nodata, lab, range(1, n + 1))
        ocean = lab == (int(np.argmax(sizes)) + 1)
        # distance_transform_edt measures distance to the nearest zero, so invert
        dist_px = ndimage.distance_transform_edt(~ocean)
        _write_like(ds, dist_px * res / 1000.0, dst_path)
        return float(ocean.mean())


def elevation_mosaic(ecostress_dir, dst_path) -> tuple:
    """One DEM for the whole box, mosaicked from the ECOSTRESS `height` layers.

    ECOSTRESS ships terrain height alongside every granule, so the elevation feature
    needs no external DEM download. One file per MGRS tile is enough - the terrain
    does not change between overpasses - and the median across the chosen files
    fills gaps left by per-granule nodata.
    """
    from collections import defaultdict

    from rasterio.merge import merge

    per_tile = defaultdict(list)
    for p in sorted(Path(ecostress_dir).rglob("*_height.tif")):
        parts = p.name.split("_")
        if len(parts) > 5:
            per_tile[parts[5]].append(p)
    if not per_tile:
        raise FileNotFoundError(f"No *_height.tif under {ecostress_dir}")

    # a handful of granules per tile, so a nodata hole in one overpass is covered
    picks = [p for files in per_tile.values() for p in files[:5]]
    srcs = [rasterio.open(p) for p in picks]
    try:
        data, transform = merge(srcs, method="max", nodata=np.nan)
        meta = srcs[0].meta.copy()
        meta.update(height=data.shape[1], width=data.shape[2], transform=transform,
                    dtype="float32", count=1, nodata=np.nan, compress="deflate", tiled=True)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **meta) as out:
            out.write(data[0].astype("float32"), 1)
    finally:
        for s in srcs:
            s.close()
    return len(per_tile), len(picks), data.shape[1:]


def build_all(cfg) -> dict:
    """Write every derived raster named in config and report what was produced."""
    r = cfg["features"]["radius_m"]
    out = {}
    for key, src_key in (("canopy_nbr_tif", "canopy_tif"), ("impervious_nbr_tif", "impervious_tif")):
        dst = cfg.path("data", key)
        size = neighbourhood_mean(cfg.path("data", src_key), dst, r)
        out[key] = f"{dst.name} ({size}x{size} window)"
        log.info("  %s -> %s (%dx%d px window)", src_key, dst.name, size, size)
    dst = cfg.path("data", "dist_coast_tif")
    frac = distance_to_coast(cfg.path("data", "impervious_tif"), dst)
    out["dist_coast_tif"] = f"{dst.name} (ocean = {frac:.1%} of grid)"
    log.info("  coast distance -> %s (ocean is %.1f%% of the grid)", dst.name, 100 * frac)

    dst = cfg.path("data", "elevation_tif")
    ntiles, nfiles, shape = elevation_mosaic(cfg.path("data", "ecostress_dir"), dst)
    out["elevation_tif"] = f"{dst.name} ({ntiles} tiles / {nfiles} granules, {shape[1]}x{shape[0]})"
    log.info("  elevation -> %s (%d tiles, %d granules)", dst.name, ntiles, nfiles)
    return out
