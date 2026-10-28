"""LST models, the cross-validation splits used to judge them, and model I/O.

Two targets are supported:
  lst   - absolute surface temperature (deg C)
  dlst  - that pixel minus its scene mean, i.e. the *spatial* anomaly

The distinction matters. Most of the variance in the raw LST column is between
scenes (which day, how hot it was), not within one scene. Routing only ever
compares paths that are walked at the same moment, so the quantity the router
actually needs is the within-scene anomaly. Scoring a model on `lst` with a split
that lets the same date appear in train and test therefore flatters it badly.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.tree import DecisionTreeRegressor

from .config import Config
from .dataset import FEATURES, TARGET, features_for

TARGETS = {"lst": "lst_c", "dlst": "dlst_c"}
SPLITS = ("random", "spatial", "temporal", "spatial_temporal")


def spatial_blocks(df: pd.DataFrame, size_deg: float) -> np.ndarray:
    """Block id per row so nearby pixels never sit on both sides of the split."""
    bx = np.floor(df["lon"].to_numpy() / size_deg).astype(int)
    by = np.floor(df["lat"].to_numpy() / size_deg).astype(int)
    return bx * 100_000 + by


def _scores(y, p) -> dict:
    return {
        "rmse_c": float(np.sqrt(mean_squared_error(y, p))),
        "mae_c": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)),
    }


def iter_folds(df: pd.DataFrame, split: str, cfg: Config, n_splits: int = 3):
    """Yield (train_idx, test_idx) for one of the four evaluation designs."""
    seed = cfg["preprocess"]["random_seed"]
    block = cfg["model"]["block_size_deg"]

    if split == "random":
        yield from KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df)
    elif split == "spatial":
        yield from GroupKFold(n_splits=n_splits).split(df, groups=spatial_blocks(df, block))
    elif split == "temporal":
        yield from GroupKFold(n_splits=n_splits).split(df, groups=df["date"].to_numpy())
    elif split == "spatial_temporal":
        # Hold out blocks and dates at once: a test row must be in an unseen block AND
        # on an unseen date, and a train row in neither. Rows that satisfy only one of
        # the two are dropped from that fold, which is what makes the split strict.
        blocks = spatial_blocks(df, block)
        dates = df["date"].to_numpy()
        ub, ud = np.unique(blocks), np.unique(dates)
        rng = np.random.default_rng(seed)
        bfold = dict(zip(rng.permutation(ub), np.arange(len(ub)) % n_splits))
        dfold = dict(zip(rng.permutation(ud), np.arange(len(ud)) % n_splits))
        bf = np.array([bfold[b] for b in blocks])
        df_ = np.array([dfold[d] for d in dates])
        for k in range(n_splits):
            te = np.nonzero((bf == k) & (df_ == k))[0]
            tr = np.nonzero((bf != k) & (df_ != k))[0]
            if len(te) and len(tr):
                yield tr, te
    else:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")


def model_factories(cfg: Config) -> dict:
    """Name -> zero-arg constructor. Kept small so a run finishes in minutes."""
    seed = cfg["preprocess"]["random_seed"]
    m = cfg["model"]
    return {
        "Linear": lambda: LinearRegression(),
        "Ridge": lambda: Ridge(alpha=1.0, random_state=None),
        "DecisionTree": lambda: DecisionTreeRegressor(
            max_depth=m.get("dt_max_depth", 12), min_samples_leaf=20, random_state=seed),
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=m.get("grid_n_estimators", 150), min_samples_leaf=m["min_samples_leaf"],
            max_depth=m.get("max_depth"), max_features=m["max_features"], n_jobs=-1, random_state=seed),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            learning_rate=m.get("hgb_learning_rate", 0.1), max_depth=m.get("hgb_max_depth", 8),
            max_iter=m.get("hgb_max_iter", 300), random_state=seed),
    }


def cross_validate(make_model, df, features, target_col, split, cfg, n_splits=3) -> dict:
    """Fit/score the model on each fold; report mean and spread across folds."""
    rows = []
    for tr, te in iter_folds(df, split, cfg, n_splits):
        mdl = make_model().fit(df.iloc[tr][features], df.iloc[tr][target_col])
        rows.append(_scores(df.iloc[te][target_col], mdl.predict(df.iloc[te][features])))
    if not rows:
        return {}
    out = {"n_folds": len(rows)}
    for k in ("rmse_c", "mae_c", "r2"):
        v = np.array([r[k] for r in rows])
        out[f"{k}_mean"], out[f"{k}_std"] = float(v.mean()), float(v.std())
    return out


def train(df: pd.DataFrame, cfg: Config):
    """Fit the model that ships with the app, on the configured target and features."""
    m = cfg["model"]
    seed = cfg["preprocess"]["random_seed"]
    features = features_for(cfg)
    target_key = m.get("target", "lst")
    target_col = TARGETS[target_key]
    split = m.get("eval_split", "spatial")

    rf_scores = cross_validate(model_factories(cfg)["RandomForest"], df, features,
                               target_col, split, cfg, m.get("n_folds", 3))
    lin_scores = cross_validate(model_factories(cfg)["Linear"], df, features,
                                target_col, split, cfg, m.get("n_folds", 3))

    # Final model: trained on everything, and capped in size so the artefact stays
    # small enough to keep next to the code (the unconstrained forest was >1 GB).
    rf = RandomForestRegressor(
        n_estimators=m["n_estimators"],
        min_samples_leaf=m["min_samples_leaf"],
        max_depth=m.get("max_depth"),
        max_features=m["max_features"],
        n_jobs=-1,
        random_state=seed,
    ).fit(df[features], df[target_col])

    metrics = {
        "target": target_key,
        "target_column": target_col,
        "eval_split": split,
        "features": features,
        "n_rows": int(len(df)),
        "n_granules": int(df["granule"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "random_forest": rf_scores,
        "linear_baseline": lin_scores,
        "feature_importance": dict(
            sorted(zip(features, map(float, rf.feature_importances_)), key=lambda kv: -kv[1])
        ),
    }
    return rf, metrics


def save(model, metrics: dict, cfg: Config) -> None:
    mp, jp = cfg.path("model", "path"), cfg.path("model", "metrics_path")
    mp.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": metrics["features"],
                 "target": metrics["target"]}, mp, compress=3)
    jp.write_text(json.dumps(metrics, indent=2))


def load(cfg: Config):
    """Return (model, features, target_key) for the trained artefact."""
    p: Path = cfg.path("model", "path")
    if not p.exists():
        raise FileNotFoundError(f"No trained model at {p}. Run: python -m scripts.train")
    b = joblib.load(p)
    return b["model"], b.get("features", FEATURES), b.get("target", "lst")
