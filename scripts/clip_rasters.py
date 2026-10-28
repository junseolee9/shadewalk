"""Task 0: clip the CONUS-wide NLCD rasters down to the LA box.

    python -m scripts.clip_rasters

The shipped impervious.tif covers the whole contiguous US (~1 GB), which makes every
read slow. This writes a LA-only copy next to it and leaves the original untouched.
Also prints CRS / resolution / nodata / value range so the inputs can be checked.
"""
import argparse

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from shadewalk.config import load_config

PAD_DEG = 0.05  # a little slack so edge pixels near the bbox are still available


def describe(path, label):
    with rasterio.open(path) as ds:
        a = ds.read(1)
        valid = a[a != ds.nodata] if ds.nodata is not None else a.ravel()
        valid = valid[(valid >= 0) & (valid <= 100)]
        print(f"[{label}] {path}")
        print(f"   CRS         {ds.crs.to_string()}")
        print(f"   size / res  {ds.width} x {ds.height} @ {ds.res[0]:g} m")
        print(f"   nodata      {ds.nodata}   dtype {ds.dtypes[0]}")
        if valid.size:
            print(f"   values      min {valid.min()} max {valid.max()} "
                  f"mean {valid.mean():.1f}  (expect 0-100 %)")
        else:
            print("   values      no pixels inside 0-100 -- check the product type")


def clip(src_path, dst_path, bbox):
    with rasterio.open(src_path) as ds:
        # the LA box is in lon/lat; the NLCD rasters are in Albers metres
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", ds.crs,
            bbox.min_lon - PAD_DEG, bbox.min_lat - PAD_DEG,
            bbox.max_lon + PAD_DEG, bbox.max_lat + PAD_DEG,
        )
        win = from_bounds(left, bottom, right, top, ds.transform).round_offsets().round_lengths()
        data = ds.read(1, window=win)
        meta = ds.meta.copy()
        meta.update(height=data.shape[0], width=data.shape[1],
                    transform=ds.window_transform(win), compress="deflate", tiled=True)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **meta) as out:
        out.write(data, 1)
    return data.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)

    src = cfg.path("data", "impervious_tif_full")
    dst = cfg.path("data", "impervious_tif")
    if dst.exists():
        print(f"{dst} already exists; re-clipping.")
    h, w = clip(src, dst, cfg.bbox)
    print(f"Clipped {src.name} -> {dst} ({w} x {h}, "
          f"{dst.stat().st_size / 1e6:.1f} MB from {src.stat().st_size / 1e6:.0f} MB)\n")

    describe(cfg.path("data", "canopy_tif"), "canopy")
    print()
    describe(dst, "impervious (clipped)")


if __name__ == "__main__":
    main()
