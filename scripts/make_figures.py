"""Every figure in the report, regenerated from the saved tables and the training CSV.

    python -m scripts.make_figures

Run after build_dataset, train, experiments and (for the route maps) evaluate_routes.
Figures whose inputs are missing are skipped with a note rather than crashing.
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from shadewalk.config import load_config  # noqa: E402
from shadewalk.dataset import features_for  # noqa: E402

log = logging.getLogger(__name__)

# categorical slots and a one-hue sequential ramp, used consistently across figures
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#3987e5", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("bu_rd", ["#184f95", "#86b6ef", "#f0efec", "#ec835a", "#d03b3b"])
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def save(fig, out: Path, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info("  %s", name)


def fig_lst_distribution(df, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].hist(df["lst_c"], bins=80, color=C[0], edgecolor="none")
    ax[0].set_xlabel("land surface temperature (°C)")
    ax[0].set_ylabel("pixels")
    ax[0].set_title("Absolute LST")
    ax[1].hist(df["dlst_c"], bins=80, color=C[2], edgecolor="none")
    ax[1].set_xlabel("ΔLST = LST − scene mean (°C)")
    ax[1].set_title("Within-scene anomaly")
    save(fig, out, "lst_distribution.png")


def fig_scene_means(df, out):
    g = (df.groupby("granule")
           .agg(mean_lst=("lst_c", "mean"), hour=("hour", "first"), date=("date", "first"))
           .reset_index())
    g["date"] = pd.to_datetime(g["date"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].scatter(g["date"], g["mean_lst"], s=28, color=C[0])
    ax[0].set_xlabel("date")
    ax[0].set_ylabel("scene mean LST (°C)")
    ax[0].set_title("Scene mean by date")
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].scatter(g["hour"], g["mean_lst"], s=28, color=C[1])
    ax[1].set_xlabel("local hour of overpass")
    ax[1].set_title("Scene mean by time of day")
    save(fig, out, "scene_means.png")


def fig_variance_decomposition(df, out):
    """The diagnosis that motivated the ΔLST target."""
    between = df.groupby("granule")["lst_c"].transform("mean")
    tot = df["lst_c"].var()
    shares = [100 * between.var() / tot, 100 * (df["lst_c"] - between).var() / tot]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars = ax.barh(["between scenes\n(day / weather)", "within a scene\n(what routing needs)"],
                   shares, color=[C[1], C[2]], height=0.55)
    for b, s in zip(bars, shares):
        ax.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2, f"{s:.1f}%",
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of total LST variance (%)")
    ax.set_title("Where the temperature variance lives")
    save(fig, out, "variance_decomposition.png")


def fig_landcover_scatter(df, out):
    s = df.sample(min(len(df), 25000), random_state=0)
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.4))
    for j, (col, label) in enumerate((("canopy_100m", "tree canopy within 100 m (%)"),
                                      ("impervious_100m", "impervious within 100 m (%)"))):
        for i, (ycol, ylab) in enumerate((("lst_c", "LST (°C)"), ("dlst_c", "ΔLST (°C)"))):
            a = ax[i][j]
            a.scatter(s[col], s[ycol], s=3, alpha=0.08, color=C[j], edgecolors="none")
            b = s.groupby(pd.cut(s[col], np.arange(0, 101, 5)), observed=False)[ycol].mean()
            centres = [iv.mid for iv in b.index]
            a.plot(centres, b.values, color="#0b0b0b", linewidth=2, label="binned mean")
            a.set_xlabel(label)
            a.set_ylabel(ylab)
            a.set_title(f"r = {s[col].corr(s[ycol]):+.3f}", fontsize=9)
            if i == 0 and j == 0:
                a.legend(fontsize=8)
    save(fig, out, "landcover_vs_lst.png")


def fig_correlation(df, features, out):
    cols = [c for c in features if c in df] + ["lst_c", "dlst_c"]
    m = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(m, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{m.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(m.iloc[i, j]) > 0.55 else "#0b0b0b")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature correlation")
    save(fig, out, "correlation_matrix.png")


def fig_boxplots(df, features, out):
    cols = [c for c in features if c in df]
    raw = df[cols]
    std = (raw - raw.mean()) / raw.std()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, data, title in ((ax[0], raw, "Raw features"), (ax[1], std, "Standardised features")):
        bp = a.boxplot([data[c].dropna() for c in cols], showfliers=False,
                       patch_artist=True, medianprops={"color": "#0b0b0b"})
        a.set_xticks(range(1, len(cols) + 1), cols)
        for patch, c in zip(bp["boxes"], C * 3):
            patch.set_facecolor(c)
            patch.set_alpha(0.65)
        a.set_title(title)
        a.tick_params(axis="x", rotation=55)
    save(fig, out, "feature_boxplots.png")


def fig_scene_maps(df, out):
    """One scene: observed LST, canopy, and the anomaly the model is asked to learn."""
    key = df.groupby("granule").size().idxmax()
    s = df[df["granule"] == key]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, col, cmap, label in ((ax[0], "lst_c", SEQ, "LST (°C)"),
                                (ax[1], "canopy_100m", SEQ, "canopy within 100 m (%)"),
                                (ax[2], "dlst_c", DIV, "ΔLST (°C)")):
        v = s[col]
        lim = max(abs(v.min()), abs(v.max())) if col == "dlst_c" else None
        sc = a.scatter(s["lon"], s["lat"], c=v, s=2.4, cmap=cmap, edgecolors="none",
                       vmin=-lim if lim else None, vmax=lim if lim else None)
        fig.colorbar(sc, ax=a, shrink=0.82, label=label)
        a.set_xlabel("longitude")
        a.grid(alpha=0.2)
    ax[0].set_ylabel("latitude")
    fig.suptitle(f"Scene {s['date'].iloc[0]} at {s['hour'].iloc[0]:.1f} h local", y=1.02, fontsize=10)
    save(fig, out, "scene_maps.png")


def fig_predicted_map(df, cfg, out):
    from shadewalk import model as M
    try:
        mdl, feats, target = M.load(cfg)
    except FileNotFoundError:
        log.info("  (skip predicted_map: no trained model)")
        return
    key = df.groupby("granule").size().idxmax()
    s = df[df["granule"] == key].copy()
    s["pred"] = mdl.predict(s[feats])
    truth_col = "dlst_c" if target == "dlst" else "lst_c"
    lim = float(np.nanpercentile(np.abs(s[truth_col]), 99))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, col, title in ((ax[0], truth_col, "observed"), (ax[1], "pred", "predicted")):
        sc = a.scatter(s["lon"], s["lat"], c=s[col], s=2.4, cmap=DIV, vmin=-lim, vmax=lim,
                       edgecolors="none")
        fig.colorbar(sc, ax=a, shrink=0.82, label=f"{'Δ' if target == 'dlst' else ''}LST (°C)")
        a.set_title(title)
        a.set_xlabel("longitude")
    ax[0].set_ylabel("latitude")
    save(fig, out, "predicted_map.png")


def fig_actual_vs_predicted(df, cfg, out):
    from sklearn.metrics import r2_score

    from shadewalk import model as M
    try:
        mdl, feats, target = M.load(cfg)
    except FileNotFoundError:
        log.info("  (skip actual_vs_predicted: no trained model)")
        return
    tr, te = next(M.iter_folds(df, cfg["model"].get("eval_split", "spatial_temporal"), cfg, 3))
    s = df.iloc[te]
    s = s.sample(min(len(s), 20000), random_state=0)
    col = "dlst_c" if target == "dlst" else "lst_c"
    p = mdl.predict(s[feats])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.scatter(s[col], p, s=3, alpha=0.1, color=C[0], edgecolors="none")
    lo, hi = float(min(s[col].min(), p.min())), float(max(s[col].max(), p.max()))
    ax.plot([lo, hi], [lo, hi], color="#0b0b0b", linewidth=1.2, linestyle="--", label="y = x")
    ax.set_xlabel(f"observed {'ΔLST' if target == 'dlst' else 'LST'} (°C)")
    ax.set_ylabel("predicted (°C)")
    ax.set_title(f"Held-out fold, R² = {r2_score(s[col], p):.3f}")
    ax.legend(fontsize=8)
    save(fig, out, "actual_vs_predicted.png")


def fig_model_comparison(tables, out):
    f = tables / "model_comparison.csv"
    if not f.exists():
        log.info("  (skip model_comparison: run scripts.experiments)")
        return
    d = pd.read_csv(f)
    for target in d["target"].unique():
        sub = d[d["target"] == target]
        models = sorted(sub["model"].unique())
        splits = [s for s in ("random", "spatial", "temporal", "spatial_temporal")
                  if s in set(sub["split"])]
        fig, ax = plt.subplots(figsize=(8.4, 3.8))
        w = 0.8 / len(models)
        x = np.arange(len(splits))
        for i, m in enumerate(models):
            vals = [sub[(sub.split == s) & (sub.model == m)]["r2_mean"].mean() for s in splits]
            errs = [sub[(sub.split == s) & (sub.model == m)]["r2_std"].mean() for s in splits]
            ax.bar(x + i * w - 0.4 + w / 2, vals, w, yerr=errs, capsize=2,
                   label=m, color=C[i % len(C)])
        ax.set_xticks(x, splits, rotation=12)
        ax.axhline(0, color="#0b0b0b", linewidth=0.8)
        ax.set_ylabel("R² (mean ± sd over folds)")
        ax.set_title(f"target = {target}")
        ax.legend(fontsize=7.5, ncol=3)
        save(fig, out, f"model_comparison_{target}.png")


def fig_importance(tables, out):
    f = tables / "importance.csv"
    if not f.exists():
        log.info("  (skip importance: run scripts.experiments)")
        return
    d = pd.read_csv(f).sort_values("permutation_mean")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.barh(y - 0.19, d["impurity"], 0.36, label="impurity (train)", color=C[3])
    ax.barh(y + 0.19, d["permutation_mean"], 0.36, xerr=d["permutation_std"], capsize=2,
            label="permutation (test)", color=C[0])
    ax.set_yticks(y, d["feature"])
    ax.set_xlabel("importance")
    ax.set_title("Feature importance, two ways")
    ax.legend(fontsize=8)
    save(fig, out, "feature_importance.png")


def fig_robustness(tables, out):
    """Per-date spread, feature ablation, and the within-scene ranking distribution."""
    need = ["lodo_cv.csv", "ablation_full.csv", "ranking_check.csv"]
    if not all((tables / f).exists() for f in need):
        log.info("  (skip robustness: run scripts.robustness)")
        return
    lodo = pd.read_csv(tables / "lodo_cv.csv")
    abl = pd.read_csv(tables / "ablation_full.csv")
    rank = pd.read_csv(tables / "ranking_check.csv")

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

    # (a) one point per held-out date, so the spread is visible rather than averaged
    for i, (name, g) in enumerate(lodo.groupby("model")):
        g = g.sort_values("held_out_date")
        ax[0].plot(range(len(g)), g["r2"], "o-", color=C[i], label=name, markersize=5, linewidth=1.4)
        ax[0].axhline(g["r2"].mean(), color=C[i], linestyle=":", linewidth=1.2)
    ax[0].axhline(0, color="#0b0b0b", linewidth=0.9)
    ax[0].set_xlabel("held-out date (chronological)")
    ax[0].set_ylabel("$R^2$ on that date")
    ax[0].set_title("Leave-one-date-out: per-date $R^2$")
    ax[0].legend(fontsize=8)

    # (b) what each feature is actually worth
    rf = abl[(abl["model"] == "RandomForest") & (abl["dropped"] != "(none: all features)")]
    rf = rf.sort_values("delta_vs_full")
    y = np.arange(len(rf))
    cols = [C[7] if d < -0.02 else C[0] if d < -0.005 else MUTED_GREY for d in rf["delta_vs_full"]]
    ax[1].barh(y, rf["delta_vs_full"], color=cols, height=0.65)
    ax[1].set_yticks(y, rf["dropped"], fontsize=8)
    ax[1].axvline(0, color="#0b0b0b", linewidth=0.9)
    ax[1].set_xlabel("$\\Delta R^2$ when the feature is removed")
    ax[1].set_title("Leave-one-feature-out (random forest)")

    # (c) ranking quality is the quantity routing needs
    for i, (name, g) in enumerate(rank.groupby("model")):
        ax[2].hist(g["pairwise_acc"], bins=18, alpha=0.6, color=C[i], label=name)
    ax[2].axvline(0.5, color="#0b0b0b", linestyle="--", linewidth=1.2, label="chance")
    ax[2].set_xlabel("pairwise accuracy within a held-out scene")
    ax[2].set_ylabel("scenes")
    ax[2].set_title("Can it order two locations correctly?")
    ax[2].legend(fontsize=8)
    save(fig, out, "robustness.png")


MUTED_GREY = "#898781"


def fig_route_examples(cfg, out):
    """Shortest vs recommended path for a couple of representative pairs."""
    import json as _json

    from shadewalk import model as M
    from shadewalk.landcover import LandCover
    from shadewalk.recommend import Recommender
    from shadewalk.routing import ORSClient, Route
    from shadewalk.scoring import decide

    ev = cfg.path("reports", "tables_dir") / "route_eval.csv"
    root = cfg.path("data", "dataset_csv").parent.parent / "cache" / "routes"
    # evaluate_routes nests caches as routes/<provider>/<sampling>/pair_NNNN.json;
    # search recursively so this does not silently go stale when that layout changes.
    by_id = {int(p.stem.split("_")[1]): p for p in root.glob("**/pair_*.json")}
    if not ev.exists() or not by_id:
        log.info("  (skip route_examples: run scripts.evaluate_routes)")
        return
    d = pd.read_csv(ev)
    # one row per pair, so the two panels show different trips rather than the same
    # trip at two hours
    picks = (d[d["changed"]].sort_values("mean_drop_c", ascending=False)
             .drop_duplicates("pair").head(2))
    if picks.empty:
        log.info("  (skip route_examples: no pair where the route changed)")
        return

    lc = LandCover.from_config(cfg)
    rec = Recommender(cfg, M.load(cfg), ORSClient(cfg), lc)
    sc = cfg["scoring"]
    fig, axes = plt.subplots(1, len(picks), figsize=(5.4 * len(picks), 4.6), squeeze=False)
    for a, (_, row) in zip(axes[0], picks.iterrows()):
        hit = by_id.get(int(row["pair"]))
        if hit is None:
            continue
        blob = _json.loads(hit.read_text())
        routes = [Route(r["name"], [tuple(c) for c in r["coords"]], r["distance_m"], r["source"])
                  for r in blob["routes"]]
        stats = [rec._stats(r, row["hour"], True) for r in routes]  # figure only; shade not needed
        dec = decide(stats, sc["weights"], sc["max_detour"], sc["min_cooling_c"])
        for r in routes:
            lon, lat = zip(*r.coords)
            if r.name == dec.chosen.name:
                a.plot(lon, lat, color=C[2], linewidth=3.2, label="recommended", zorder=3)
            elif r.name == dec.shortest.name:
                a.plot(lon, lat, color="#0b0b0b", linewidth=2, linestyle="--",
                       label="shortest", zorder=2)
            else:
                a.plot(lon, lat, color="#898781", linewidth=1.1, alpha=0.7, zorder=1)
        a.set_title(f"pair {int(row['pair'])} at {int(row['hour'])}:00 — "
                    f"{row['mean_drop_c']:.1f}°C cooler, +{row['detour_pct']:.0f}% longer",
                    fontsize=8.5)
        a.set_xlabel("longitude")
        a.legend(fontsize=8)
    axes[0][0].set_ylabel("latitude")
    lc.close()
    save(fig, out, "route_examples.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    out = cfg.path("reports", "figures_dir")
    tables = cfg.path("reports", "tables_dir")
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.path("data", "dataset_csv"))
    features = [f for f in features_for(cfg) if f in df.columns]
    log.info("Figures from %d rows -> %s", len(df), out)

    fig_lst_distribution(df, out)
    fig_scene_means(df, out)
    fig_variance_decomposition(df, out)
    fig_landcover_scatter(df, out)
    fig_correlation(df, features, out)
    fig_boxplots(df, features, out)
    fig_scene_maps(df, out)
    fig_predicted_map(df, cfg, out)
    fig_actual_vs_predicted(df, cfg, out)
    fig_model_comparison(tables, out)
    fig_importance(tables, out)
    fig_robustness(tables, out)
    fig_route_examples(cfg, out)
    print(f"\nFigures -> {out}")


if __name__ == "__main__":
    main()
