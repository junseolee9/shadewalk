"""Task 4: the model x target x split grid behind the report tables.

    python -m scripts.experiments

Writes to reports/tables/:
  features.csv            variable list with source / type / derived / coeff. of variation
  model_comparison.csv    every (target, split, model) cell with fold mean and spread
  hyperparams.csv         the small in-fold search for RF and HGB
  importance.csv          impurity and permutation importance for the best model
  hour_ablation.csv       dLST with and without the hour column
  model_size.csv          accuracy vs file size for the shipped forest
"""
import argparse
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from shadewalk.config import load_config
from shadewalk.dataset import features_for
from shadewalk.model import SPLITS, TARGETS, cross_validate, iter_folds, model_factories

log = logging.getLogger(__name__)

# where each variable comes from, for the report's variable table
SOURCES = {
    "lat": ("ECOSTRESS", "float", "No"),
    "lon": ("ECOSTRESS", "float", "No"),
    "hour": ("ECOSTRESS", "float", "Yes"),
    "canopy_pct": ("NLCD", "float", "No"),
    "impervious_pct": ("NLCD", "float", "No"),
    "canopy_100m": ("NLCD", "float", "Yes"),
    "impervious_100m": ("NLCD", "float", "Yes"),
    "elevation_m": ("ECOSTRESS", "float", "Yes"),
    "dist_coast_km": ("NLCD", "float", "Yes"),
}


def subsample(df, n, seed):
    """Keep the grid affordable. Sampling is stratified by granule so every scene stays."""
    if len(df) <= n:
        return df
    per = max(1, n // df["granule"].nunique())
    return (df.groupby("granule", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per), random_state=seed))
              .reset_index(drop=True))


def features_table(df, features, out_dir):
    rows = []
    for f in features:
        src, typ, derived = SOURCES.get(f, ("-", "float", "-"))
        mu, sd = df[f].mean(), df[f].std()
        rows.append({"feature": f, "source": src, "type": typ, "derived": derived,
                     "mean": round(float(mu), 3), "std": round(float(sd), 3),
                     "coeff_variation": round(float(sd / mu), 3) if mu else np.nan,
                     "corr_with_dlst": round(float(df[f].corr(df["dlst_c"])), 3)})
    t = pd.DataFrame(rows)
    t.to_csv(out_dir / "features.csv", index=False)
    return t


def tune(name, make, df, features, target_col, cfg, grid):
    """Small search evaluated inside the training folds of the spatial+temporal split.

    One search per model is reused across the other splits: a full nested search over
    every cell would multiply the runtime without changing the ranking we report.
    """
    rows = []
    for params in grid:
        t0 = time.time()
        sc = cross_validate(lambda: make().set_params(**params), df, features,
                            target_col, "spatial_temporal", cfg, 2)
        rows.append({"model": name, **params, **sc, "seconds": round(time.time() - t0, 1)})
        log.info("    %s %s -> R2 %.3f", name, params, sc.get("r2_mean", float("nan")))
    best = max(rows, key=lambda r: r.get("r2_mean", -np.inf))
    return rows, {k: v for k, v in best.items() if k in grid[0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    seed = cfg["preprocess"]["random_seed"]
    out = cfg.path("reports", "tables_dir")
    out.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(cfg.path("data", "dataset_csv"))
    features = features_for(cfg)
    df = subsample(full, cfg["model"].get("grid_rows", 60000), seed)
    log.info("Grid runs on %d of %d rows (%d granules, %d dates)",
             len(df), len(full), df["granule"].nunique(), df["date"].nunique())

    log.info("\n[1/5] variable table")
    print(features_table(full, features, out).to_string(index=False))

    log.info("\n[2/5] hyper-parameter search (spatial_temporal folds)")
    tuned, hp_rows = {}, []
    rf_grid = [{"max_depth": d, "min_samples_leaf": m}
               for d in (10, 20, None) for m in (5, 50)]
    hgb_grid = [{"learning_rate": lr, "max_depth": d}
                for lr in (0.05, 0.1) for d in (4, 8)]
    for name, grid in (("RandomForest", rf_grid), ("HistGradientBoosting", hgb_grid)):
        rows, best = tune(name, model_factories(cfg)[name], df, features,
                          TARGETS["dlst"], cfg, grid)
        hp_rows += rows
        tuned[name] = best
        log.info("  best %s: %s", name, best)
    pd.DataFrame(hp_rows).to_csv(out / "hyperparams.csv", index=False)

    log.info("\n[3/5] model x target x split grid")
    rows = []
    for target_key, target_col in TARGETS.items():
        for split in SPLITS:
            for name, make in model_factories(cfg).items():
                params = tuned.get(name, {})
                t0 = time.time()
                sc = cross_validate(lambda: make().set_params(**params) if params else make(),
                                    df, features, target_col, split, cfg,
                                    cfg["model"].get("n_folds", 3))
                if not sc:
                    continue
                rows.append({"target": target_key, "split": split, "model": name,
                             **{k: round(v, 4) for k, v in sc.items()},
                             "seconds": round(time.time() - t0, 1)})
                log.info("  %-5s %-17s %-20s R2 %.3f +- %.3f  RMSE %.2f",
                         target_key, split, name, sc["r2_mean"], sc["r2_std"], sc["rmse_c_mean"])
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "model_comparison.csv", index=False)

    log.info("\n[4/5] hour ablation on dLST")
    abl = []
    for use_hour in (True, False):
        feats = features if use_hour else [f for f in features if f != "hour"]
        for split in ("spatial", "temporal", "spatial_temporal"):
            for name in ("Linear", "RandomForest"):
                make = model_factories(cfg)[name]
                params = tuned.get(name, {})
                sc = cross_validate(lambda: make().set_params(**params) if params else make(),
                                    df, feats, TARGETS["dlst"], split, cfg, 3)
                abl.append({"hour_included": use_hour, "split": split, "model": name,
                            **{k: round(v, 4) for k, v in sc.items()}})
                log.info("  hour=%-5s %-17s %-13s R2 %.3f", use_hour, split, name, sc["r2_mean"])
    pd.DataFrame(abl).to_csv(out / "hour_ablation.csv", index=False)

    log.info("\n[5/5] importance + model size")
    tr, te = next(iter_folds(df, "spatial_temporal", cfg, 3))
    rf = RandomForestRegressor(n_estimators=cfg["model"]["grid_n_estimators"], n_jobs=-1,
                               random_state=seed, **tuned["RandomForest"])
    rf.fit(df.iloc[tr][features], df.iloc[tr][TARGETS["dlst"]])
    perm = permutation_importance(rf, df.iloc[te][features], df.iloc[te][TARGETS["dlst"]],
                                  n_repeats=5, random_state=seed, n_jobs=-1)
    imp = pd.DataFrame({"feature": features,
                        "impurity": rf.feature_importances_,
                        "permutation_mean": perm.importances_mean,
                        "permutation_std": perm.importances_std}
                       ).sort_values("permutation_mean", ascending=False)
    imp.round(4).to_csv(out / "importance.csv", index=False)
    print(imp.round(4).to_string(index=False))

    sizes = []
    tmp = out.parent / "_size_probe.joblib"
    for depth, n_est in ((None, 200), (30, 200), (20, 200), (15, 200), (20, 100)):
        m = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                  min_samples_leaf=cfg["model"]["min_samples_leaf"],
                                  max_features=cfg["model"]["max_features"],
                                  n_jobs=-1, random_state=seed)
        m.fit(df.iloc[tr][features], df.iloc[tr][TARGETS["dlst"]])
        joblib.dump({"model": m}, tmp, compress=3)
        from sklearn.metrics import r2_score
        sizes.append({"max_depth": depth, "n_estimators": n_est,
                      "mb": round(tmp.stat().st_size / 1e6, 1),
                      "r2": round(float(r2_score(df.iloc[te][TARGETS["dlst"]],
                                                 m.predict(df.iloc[te][features]))), 4)})
        log.info("  depth=%-4s n=%-3d  %5.1f MB  R2 %.3f",
                 depth, n_est, sizes[-1]["mb"], sizes[-1]["r2"])
    tmp.unlink(missing_ok=True)
    pd.DataFrame(sizes).to_csv(out / "model_size.csv", index=False)

    print(f"\nTables -> {out}")


if __name__ == "__main__":
    main()
