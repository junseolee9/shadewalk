"""Build the training table: daytime ECOSTRESS pixels joined with land cover."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config
from .ecostress import find_granules, granule_pixels
from .landcover import LandCover

log = logging.getLogger(__name__)

FEATURES = ["lat", "lon", "canopy_pct", "impervious_pct", "hour"]  # minimum set, always present
TARGET = "lst_c"


def features_for(cfg: Config) -> list:
    """Feature columns for this run. Configurable so ablations do not need code edits."""
    return list((cfg.get("features") or {}).get("columns") or FEATURES)


def is_daytime(hour: float, cfg: Config) -> bool:
    d = cfg["daytime"]
    return d["start_hour"] <= hour < d["end_hour"]


def build_dataset(cfg: Config) -> pd.DataFrame:
    pre = cfg["preprocess"]
    rng = np.random.default_rng(pre["random_seed"])
    tz = cfg["timezone"]

    granules = find_granules(cfg.path("data", "ecostress_dir"))
    if not granules:
        raise FileNotFoundError(f"No ECOSTRESS LST tifs in {cfg.path('data', 'ecostress_dir')}")

    day = [g for g in granules if is_daytime(g.local_time(tz).hour + g.local_time(tz).minute / 60, cfg)]
    log.info("%d granules found, %d in the daytime window", len(granules), len(day))

    parts, per_granule, skipped_few, skipped_frac = [], [], 0, 0
    for i, g in enumerate(day, 1):
        df, st = granule_pixels(
            g, cfg.bbox, tz, pre["keep_qc_values"], tuple(pre["lst_valid_range_c"]),
            pre["samples_per_granule"], rng,
            max_lst_err_k=pre.get("max_lst_err_k", float("inf")),
            robust_z=pre.get("robust_z", float("inf")),
        )
        st["kept"] = False
        if st["valid_fraction"] < pre.get("min_valid_fraction", 0.0):
            skipped_frac += 1          # mostly cloud over the city: the scene tells us nothing
        elif len(df) < pre["min_valid_pixels"]:
            skipped_few += 1
        else:
            st["kept"] = True
            parts.append(df)
        per_granule.append(st)
        if i % 20 == 0:
            log.info("  processed %d/%d granules", i, len(day))
    if not parts:
        raise RuntimeError("Every daytime granule was empty after cloud/QC masking.")
    log.info("kept %d granules, skipped %d (low valid fraction) + %d (too few pixels)",
             len(parts), skipped_frac, skipped_few)

    data = pd.concat(parts, ignore_index=True)
    lc = LandCover.from_config(cfg)
    try:
        for k, v in lc.features(data["lon"].to_numpy(), data["lat"].to_numpy()).items():
            data[k] = v
    finally:
        lc.close()

    before = len(data)
    required = [c for c in set(features_for(cfg)) | set(FEATURES) if c in data.columns]
    data = data.dropna(subset=required + [TARGET]).reset_index(drop=True)
    log.info("%d rows, dropped %d with missing land cover", len(data), before - len(data))

    # Scene-relative target: LST minus that scene's mean. Routing compares paths within
    # one scene, so this is the part of the signal the route step actually uses.
    data["scene_mean_c"] = data.groupby("granule")["lst_c"].transform("mean")
    data["dlst_c"] = data["lst_c"] - data["scene_mean_c"]

    _write_funnel(cfg, granules, day, per_granule, skipped_frac, skipped_few, before, len(data))
    return data


def _write_funnel(cfg, granules, day, per_granule, skipped_frac, skipped_few, rows_before, rows_after):
    """Report table: how many granules/pixels survive each cleaning stage."""
    st = pd.DataFrame(per_granule)
    kept = st[st["kept"]]
    stages = [
        ("granules downloaded", len(granules), ""),
        ("granules in daytime window", len(day), f"{cfg['daytime']['start_hour']}-{cfg['daytime']['end_hour']} h local"),
        ("granules after valid-fraction filter", len(day) - skipped_frac,
         f">= {cfg['preprocess'].get('min_valid_fraction')} of LA-box pixels valid"),
        ("granules used", len(kept), f">= {cfg['preprocess']['min_valid_pixels']} sampled pixels"),
        ("pixels inside LA box", int(st["px_in_box"].sum()), "before any masking"),
        ("after LST range check", int(st["px_lst_range"].sum()),
         "{} to {} C".format(*cfg["preprocess"]["lst_valid_range_c"])),
        ("after cloud mask", int(st["px_cloud"].sum()), "cloud == 0"),
        ("after water mask", int(st["px_water"].sum()), "water == 0"),
        ("after QC mask", int(st["px_qc"].sum()),
         "QC bits 0-1 in {}".format("/".join(map(str, cfg["preprocess"]["keep_qc_values"])))),
        ("after LST_err cut", int(st["px_lst_err"].sum()), f"<= {cfg['preprocess'].get('max_lst_err_k')} K"),
        ("after robust MAD cut", int(st["px_robust"].sum()), f"{cfg['preprocess'].get('robust_z')} x MAD"),
        ("rows sampled", rows_before, f"<= {cfg['preprocess']['samples_per_granule']} per granule"),
        ("rows after land-cover join", rows_after, "canopy + impervious present"),
    ]
    out = pd.DataFrame(stages, columns=["stage", "count", "criterion"])
    path = cfg.path("reports", "tables_dir") / "data_funnel.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    st.to_csv(path.parent / "granule_stats.csv", index=False)
    log.info("Funnel table -> %s", path)
