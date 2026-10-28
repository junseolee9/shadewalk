"""Robustness experiments that back up (or qualify) the claims in the report.

    python -m scripts.robustness

Four things the first round of experiments left open:

  1. leave-one-date-out CV with a bootstrap CI, because 18 dates in 3 folds gives a
     fold spread too wide to quote a single R2 from;
  2. a properly nested hyper-parameter search, so the model comparison is not quietly
     tuned on the same folds it is scored on;
  3. a leave-one-out ablation over every feature, so `hour` is not the only variable
     tested for being a scene identifier -- `dist_coast_km` is equally suspect;
  4. a within-scene *ranking* check, which is what route recommendation actually needs:
     the router never needs a calibrated degree value, only the correct ordering.

Writes reports/tables/: loDo_cv.csv, nested_cv.csv, ablation_full.csv, ranking_check.csv
"""
import argparse
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

from shadewalk.config import load_config
from shadewalk.dataset import features_for
from shadewalk.model import TARGETS, _scores, iter_folds, model_factories, spatial_blocks

log = logging.getLogger(__name__)


def subsample(df, n, seed):
    if len(df) <= n:
        return df
    per = max(1, n // df["granule"].nunique())
    return (df.groupby("granule", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per), random_state=seed))
              .reset_index(drop=True))


# ---------------------------------------------------------------- 1. leave-one-date-out

def leave_one_date_out(df, features, cfg, model_names=("Linear", "RandomForest")):
    """Train on all dates but one, test on the held-out date. One row per (model, date).

    With 18 dates this gives 18 independent estimates instead of 3, so the spread can be
    summarised with a bootstrap interval rather than a standard deviation over 3 numbers.
    """
    dates = sorted(df["date"].unique())
    rows = []
    for name in model_names:
        make = model_factories(cfg)[name]
        for d in dates:
            te = df["date"].to_numpy() == d
            if te.sum() < 200 or (~te).sum() < 1000:
                continue
            mdl = make().fit(df.loc[~te, features], df.loc[~te, TARGETS["dlst"]])
            sc = _scores(df.loc[te, TARGETS["dlst"]], mdl.predict(df.loc[te, features]))
            rows.append({"model": name, "held_out_date": d, "n_test": int(te.sum()), **sc})
            log.info("  %-13s %s  R2 %+.3f", name, d, sc["r2"])
    return pd.DataFrame(rows)


def bootstrap_ci(values, n_boot=10000, seed=0):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- 2. nested CV

def nested_cv(df, features, cfg, grids, split="spatial_temporal", n_outer=3, n_inner=2):
    """Tune inside each outer training fold, score on the untouched outer test fold.

    The headline grid reused one tuning result across splits for speed; this checks that
    the shortcut did not change the conclusion.
    """
    rows = []
    for name, grid in grids.items():
        make = model_factories(cfg)[name]
        outer = []
        for k, (tr, te) in enumerate(iter_folds(df, split, cfg, n_outer)):
            inner_df = df.iloc[tr].reset_index(drop=True)
            best, best_r2 = None, -np.inf
            for params in grid:
                r2s = []
                for itr, ite in iter_folds(inner_df, split, cfg, n_inner):
                    m = make().set_params(**params).fit(
                        inner_df.iloc[itr][features], inner_df.iloc[itr][TARGETS["dlst"]])
                    r2s.append(r2_score(inner_df.iloc[ite][TARGETS["dlst"]],
                                        m.predict(inner_df.iloc[ite][features])))
                if r2s and np.mean(r2s) > best_r2:
                    best, best_r2 = params, float(np.mean(r2s))
            mdl = make().set_params(**(best or {})).fit(
                df.iloc[tr][features], df.iloc[tr][TARGETS["dlst"]])
            sc = _scores(df.iloc[te][TARGETS["dlst"]], mdl.predict(df.iloc[te][features]))
            outer.append(sc)
            rows.append({"model": name, "outer_fold": k, "chosen_params": str(best), **sc})
            log.info("  %-20s fold %d  params %-45s R2 %+.3f", name, k, str(best), sc["r2"])
        if outer:
            log.info("  %-20s nested mean R2 %+.3f", name, np.mean([o["r2"] for o in outer]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. full ablation

def full_ablation(df, features, cfg, split="spatial_temporal", n_splits=3):
    """Drop each feature in turn. A feature whose removal costs nothing was not helping;
    one that helps only under a lenient split was memorising the scene."""
    rows = []
    for name in ("Linear", "RandomForest"):
        make = model_factories(cfg)[name]
        base = None
        for dropped in [None] + list(features):
            feats = [f for f in features if f != dropped]
            if not feats:
                continue
            r2s = []
            for tr, te in iter_folds(df, split, cfg, n_splits):
                m = make().fit(df.iloc[tr][feats], df.iloc[tr][TARGETS["dlst"]])
                r2s.append(r2_score(df.iloc[te][TARGETS["dlst"]], m.predict(df.iloc[te][feats])))
            mean = float(np.mean(r2s))
            if dropped is None:
                base = mean
            rows.append({"model": name, "dropped": dropped or "(none: all features)",
                         "r2_mean": round(mean, 4), "r2_std": round(float(np.std(r2s)), 4),
                         "delta_vs_full": round(mean - base, 4) if base is not None else 0.0})
            log.info("  %-13s drop %-18s R2 %+.3f  (delta %+.3f)",
                     name, dropped or "nothing", mean, rows[-1]["delta_vs_full"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. ranking check

def ranking_check(df, features, cfg, split="spatial_temporal", n_splits=3, seed=0):
    """Can the model order locations within a scene, even if the degrees are off?

    Route scoring only compares candidates inside one scene, so Spearman rank correlation
    between predicted and observed anomaly -- computed per held-out scene -- is a more
    faithful measure of fitness for purpose than pooled R2. We also report pairwise
    accuracy: given two random pixels in the same scene, how often is the cooler one
    correctly identified?
    """
    rng = np.random.default_rng(seed)
    rows = []
    for name in ("Linear", "RandomForest"):
        make = model_factories(cfg)[name]
        for k, (tr, te) in enumerate(iter_folds(df, split, cfg, n_splits)):
            mdl = make().fit(df.iloc[tr][features], df.iloc[tr][TARGETS["dlst"]])
            sub = df.iloc[te].copy()
            sub["pred"] = mdl.predict(sub[features])
            for g, s in sub.groupby("granule"):
                if len(s) < 100:
                    continue
                rho = stats.spearmanr(s["pred"], s[TARGETS["dlst"]]).statistic
                a = rng.integers(0, len(s), 20000)
                b = rng.integers(0, len(s), 20000)
                ok = a != b
                p, y = s["pred"].to_numpy(), s[TARGETS["dlst"]].to_numpy()
                agree = (np.sign(p[a[ok]] - p[b[ok]]) == np.sign(y[a[ok]] - y[b[ok]])).mean()
                rows.append({"model": name, "fold": k, "granule": g, "n": len(s),
                             "spearman": round(float(rho), 4),
                             "pairwise_acc": round(float(agree), 4)})
    out = pd.DataFrame(rows)
    for name, g in out.groupby("model"):
        log.info("  %-13s median Spearman %.3f | mean pairwise accuracy %.3f",
                 name, g["spearman"].median(), g["pairwise_acc"].mean())
    return out


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
    log.info("Robustness runs on %d of %d rows (%d granules, %d dates)\n",
             len(df), len(full), df["granule"].nunique(), df["date"].nunique())

    log.info("[1/4] leave-one-date-out CV")
    lodo = leave_one_date_out(df, features, cfg)
    lodo.to_csv(out / "lodo_cv.csv", index=False)
    print()
    summary = []
    for name, g in lodo.groupby("model"):
        m, lo, hi = bootstrap_ci(g["r2"], seed=seed)
        summary.append({"model": name, "n_dates": len(g), "r2_mean": round(m, 4),
                        "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "r2_median": round(float(g["r2"].median()), 4),
                        "r2_min": round(float(g["r2"].min()), 4),
                        "r2_max": round(float(g["r2"].max()), 4)})
        log.info("  %-13s R2 %.3f  95%% CI [%.3f, %.3f]  over %d dates",
                 name, m, lo, hi, len(g))
    pd.DataFrame(summary).to_csv(out / "lodo_summary.csv", index=False)

    log.info("\n[2/4] nested hyper-parameter search")
    grids = {
        "RandomForest": [{"max_depth": d, "min_samples_leaf": m}
                         for d in (10, 20, None) for m in (5, 50)],
        "HistGradientBoosting": [{"learning_rate": lr, "max_depth": d}
                                 for lr in (0.05, 0.1) for d in (4, 8)],
    }
    nested_cv(df, features, cfg, grids).to_csv(out / "nested_cv.csv", index=False)

    log.info("\n[3/4] leave-one-feature-out ablation")
    full_ablation(df, features, cfg).to_csv(out / "ablation_full.csv", index=False)

    log.info("\n[4/4] within-scene ranking check")
    rank = ranking_check(df, features, cfg)
    rank.to_csv(out / "ranking_check.csv", index=False)

    print(f"\nTables -> {out}")


if __name__ == "__main__":
    main()
