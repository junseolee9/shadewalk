"""Task 5: does the recommender actually produce cooler walks than the shortest path?

    python -m scripts.evaluate_routes --pairs 100

Builds a fixed set of origin/destination pairs, fetches the ORS candidates once and
caches them on disk, then re-scores those cached candidates under every weighting and
detour cap. Only the first run spends API calls; the sensitivity sweep is free.

Writes reports/tables/route_eval.csv, route_sensitivity.csv and
reports/figures/tradeoff.png.
"""
import argparse
import json
import logging
import time

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from shadewalk import model  # noqa: E402
from shadewalk.config import load_config  # noqa: E402
from shadewalk.landcover import LandCover  # noqa: E402
from shadewalk.recommend import Recommender  # noqa: E402
from shadewalk.routing import FreeRouteClient, ORSClient, Route, haversine  # noqa: E402
from shadewalk.scoring import decide  # noqa: E402

log = logging.getLogger(__name__)
EVAL_HOURS = (10, 13, 16)


# Canopy strata for the stratified sample. Uniform sampling over built-up land draws
# mostly from the uniformly paved interior of the basin, where no candidate route differs
# much from any other -- which averages the interesting cases away.
STRATA = (("low", 0.0, 6.0), ("mid", 6.0, 18.0), ("high", 18.0, 101.0))


def _point_pool(lc, b, lo, hi, rng, want, stratified):
    """Draw random built-up points until each stratum has `want` candidates on hand."""
    pool = {k: [] for k in want}
    tries = 0
    while any(len(pool[k]) < want[k] for k in want) and tries < 2000:
        tries += 1
        lon = rng.uniform(b.min_lon, b.max_lon, 4096)
        lat = rng.uniform(b.min_lat, b.max_lat, 4096)
        f = lc.features(lon, lat)
        imp, can = f["impervious_pct"], f.get("canopy_100m", f["canopy_pct"])
        ok = np.nonzero(np.isfinite(imp) & (imp > 10) & np.isfinite(can))[0]
        for i in ok:
            name = ("all" if not stratified else
                    next((s for s, lo2, hi2 in STRATA if lo2 <= can[i] < hi2), None))
            if name is not None and len(pool[name]) < want[name]:
                pool[name].append((float(lon[i]), float(lat[i]), float(can[i])))
    return pool


def fill_strata_live(cfg, router, cache, target_n, seed, min_km, max_km, sleep_s,
                     stratified=True):
    """Fetch pairs stratum by stratum, never moving on until `target_n` of them have a
    live, cached route. A pair that turns out unroutable (bad coordinates, no footpath
    in that spot) is simply discarded and replaced with a fresh random draw from the
    same stratum -- so every run ends with exactly `target_n` per stratum rather than
    "however many happened to survive".
    """
    rng = np.random.default_rng(seed)
    b = cfg.bbox
    lc = LandCover.from_config(cfg)
    names = [s[0] for s in STRATA] if stratified else ["all"]
    pool = _point_pool(lc, b, 0, 0, rng, {n: 40 * target_n for n in names}, stratified)
    lc.close()

    rows, pair_id = [], 0
    for name in names:
        pts = pool[name]
        log.info("stratum %-5s: pool of %d candidate points, filling to %d routable pairs",
                 name, len(pts), target_n)
        success, attempts, tried = 0, 0, set()
        max_attempts = max(2000, 300 * target_n)
        while success < target_n and attempts < max_attempts:
            attempts += 1
            if len(pts) < 2:
                break
            i, j = rng.integers(0, len(pts), 2)
            if i == j or (i, j) in tried or (j, i) in tried:
                continue
            tried.add((i, j))
            (x1, y1, c1), (x2, y2, c2) = pts[i], pts[j]
            d = float(haversine(x1, y1, x2, y2)) / 1000.0
            if not (min_km <= d <= max_km):
                continue
            # The cache key must be unique per *attempt*, not per success: reusing a
            # not-yet-incremented pair_id here means a failed attempt's cached error
            # would wrongly short-circuit the next, entirely different, coordinate pair
            # that happens to land in the same unincremented slot (confirmed: this was
            # silently capping the real hit rate to a few percent).
            row = pd.Series({"pair": pair_id, "stratum": name,
                             "start_lon": x1, "start_lat": y1, "end_lon": x2, "end_lat": y2,
                             "straight_km": round(d, 3), "mean_canopy_100m": round((c1 + c2) / 2, 2)})
            pair_id += 1
            try:
                routes = cached_candidates(router, row, cache, sleep_s)
            except QuotaExceeded as e:
                # Preserve whatever was already filled rather than losing it to an
                # unwound exception; the caller just gets a short result and logs it.
                log.warning("\nQuota exhausted mid-fill (stratum %s, %d/%d so far): %s",
                           name, success, target_n, e)
                return pd.DataFrame(rows), True
            if not routes or len(routes) < 2:
                continue  # unroutable -- discard and let the loop draw a fresh random pair
            rows.append(row.to_dict())
            success += 1
        if success < target_n:
            log.warning("  stratum %s: only %d/%d routable pairs found in %d attempts",
                       name, success, target_n, attempts)
        else:
            log.info("  stratum %s: filled (%d attempts, %.0f%% hit rate)",
                     name, attempts, 100 * success / attempts)
    return pd.DataFrame(rows), False


def make_pairs(cfg, n, seed, min_km=0.8, max_km=3.0, stratified=True):
    """Origin/destination pairs inside the LA box, saved once and reused.

    Points are rejected unless they sit on built-up land (impervious > 10 %), which keeps
    them off the ocean and out of the mountains where no pedestrian network exists. With
    `stratified`, pairs are allocated evenly across neighbourhood-canopy strata so that
    leafy areas -- where a cooler alternative can plausibly exist -- are represented rather
    than drowned out.
    """
    path = cfg.path("data", "dataset_csv").parent / (
        "route_pairs_stratified.csv" if stratified else "route_pairs.csv")
    if path.exists():
        return pd.read_csv(path)

    rng = np.random.default_rng(seed)
    b = cfg.bbox
    lc = LandCover.from_config(cfg)
    want = {s[0]: n // len(STRATA) for s in STRATA} if stratified else {"all": n}

    # Collect a pool of valid points per stratum first, then pair points *within* a
    # stratum. Pairing random points and hoping both land in the same stratum finds
    # almost no high-canopy pairs, because high-canopy built-up land is rare.
    pool = {k: [] for k in want}
    rows, tries = [], 0
    try:
        while any(len(pool[k]) < 8 * want[k] for k in want) and tries < 600:
            tries += 1
            lon = rng.uniform(b.min_lon, b.max_lon, 4096)
            lat = rng.uniform(b.min_lat, b.max_lat, 4096)
            f = lc.features(lon, lat)
            imp, can = f["impervious_pct"], f.get("canopy_100m", f["canopy_pct"])
            ok = np.nonzero(np.isfinite(imp) & (imp > 10) & np.isfinite(can))[0]
            for i in ok:
                name = ("all" if not stratified else
                        next((s for s, lo, hi in STRATA if lo <= can[i] < hi), None))
                if name is not None and len(pool[name]) < 8 * want[name]:
                    pool[name].append((float(lon[i]), float(lat[i]), float(can[i])))
        for name, quota in want.items():
            pts = pool[name]
            log.info("  stratum %-5s pool=%d, need %d pairs", name, len(pts), quota)
            made, attempts = 0, 0
            while made < quota and attempts < 60000 and len(pts) > 1:
                attempts += 1
                i, j = rng.integers(0, len(pts), 2)
                if i == j:
                    continue
                (x1, y1, c1), (x2, y2, c2) = pts[i], pts[j]
                d = float(haversine(x1, y1, x2, y2)) / 1000.0
                if not (min_km <= d <= max_km):
                    continue
                rows.append({"pair": len(rows), "stratum": name,
                             "start_lon": x1, "start_lat": y1, "end_lon": x2, "end_lat": y2,
                             "straight_km": round(d, 3),
                             "mean_canopy_100m": round((c1 + c2) / 2, 2)})
                made += 1
            if made < quota:
                log.warning("  stratum %s: only %d/%d pairs found", name, made, quota)
    finally:
        lc.close()
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    log.info("Wrote %d pairs -> %s", len(out), path)
    if stratified and len(out):
        log.info("  strata: %s", out["stratum"].value_counts().to_dict())
    return out


class QuotaExceeded(RuntimeError):
    """The ORS daily/rate quota is gone; stop rather than burn through the pair list."""


# 403/429 are explicit quota/rate-limit signals; caching them as "unroutable" would
# permanently poison the cache, so they abort the run instead. A bare 500 is NOT included:
# confirmed empirically (pair failed, two unrelated pairs immediately after succeeded) that
# freeroute.org returns a generic 500 for specific unroutable coordinates (observed near
# LAX, where the foot-walking graph has no connectivity) rather than a clean 400 -- that is
# a per-pair failure, not quota, and belongs in the normal "unroutable" cache path below.
TRANSIENT = ("403", "429", "502", "503", "504", "Quota exceeded", "rate limit")


def cached_candidates(router, pair, cache_dir, sleep_s, offline=False):
    """ORS candidates for one pair, cached as JSON so a re-run costs no API calls."""
    f = cache_dir / f"pair_{int(pair['pair']):04d}.json"
    if f.exists():
        d = json.loads(f.read_text())
        if d.get("error"):
            return None
        return [Route(r["name"], [tuple(c) for c in r["coords"]], r["distance_m"], r["source"])
                for r in d["routes"]]
    if offline:
        return None
    try:
        routes = router.candidates((pair["start_lon"], pair["start_lat"]),
                                   (pair["end_lon"], pair["end_lat"]))
    except Exception as e:
        msg = str(e)
        if any(t in msg for t in TRANSIENT):
            raise QuotaExceeded(msg[:200])
        f.write_text(json.dumps({"error": msg[:200]}))   # genuinely unroutable: remember it
        log.warning("  pair %s unroutable: %s", pair["pair"], msg[:90])
        return None
    finally:
        time.sleep(sleep_s)                      # stay inside the free ORS quota
    f.write_text(json.dumps({"routes": [{"name": r.name, "coords": r.coords,
                                         "distance_m": r.distance_m, "source": r.source}
                                        for r in routes]}))
    return routes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=1.6, help="seconds between ORS calls")
    ap.add_argument("--offline", action="store_true",
                    help="score only pairs already in the cache; make no API calls")
    ap.add_argument("--uniform", action="store_true",
                    help="uniform sampling over built-up land instead of canopy-stratified")
    ap.add_argument("--provider", choices=["ors", "freeroute"], default="ors",
                    help="directions provider -- each has its own daily quota and its own "
                         "route cache, so switching providers never reuses the wrong routes")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    seed = cfg["preprocess"]["random_seed"]
    tables = cfg.path("reports", "tables_dir")
    figures = cfg.path("reports", "figures_dir")
    cache = cfg.path("data", "dataset_csv").parent.parent / "cache" / "routes"
    for d in (tables, figures, cache):
        d.mkdir(parents=True, exist_ok=True)

    cache = cache / args.provider / ("uniform" if args.uniform else "stratified")
    cache.mkdir(parents=True, exist_ok=True)
    router = ORSClient(cfg) if args.provider == "ors" else FreeRouteClient(cfg)
    log.info("Provider: %s | cache: %s", args.provider, cache)

    if args.offline:
        pairs = make_pairs(cfg, args.pairs, seed, stratified=not args.uniform)
    else:
        # Guaranteed fill: never move to the next stratum, and never report a run short
        # of its target, because a few pairs landed somewhere unroutable. Each stratum
        # gets exactly `target_n` *routable* pairs, retrying fresh random draws until it
        # does. This replaces the old "draw N pairs once, keep whichever survive" design,
        # which could under-represent a stratum purely because of bad luck.
        n_strata = len(STRATA) if not args.uniform else 1
        target_n = max(1, args.pairs // n_strata)
        pairs, fill_quota_hit = fill_strata_live(
            cfg, router, cache, target_n, seed,
            min_km=0.8, max_km=3.0, sleep_s=args.sleep, stratified=not args.uniform)
        pairs_path = cfg.path("data", "dataset_csv").parent / (
            f"route_pairs_{args.provider}_{'uniform' if args.uniform else 'stratified'}.csv")
        pairs.to_csv(pairs_path, index=False)
        log.info("Wrote %d guaranteed-routable pairs -> %s", len(pairs), pairs_path)
        if fill_quota_hit and pairs.empty:
            log.warning("Quota hit before any pair was filled; nothing to score. "
                       "Re-run the same command once the quota resets.")
            return
    lc = LandCover.from_config(cfg)
    rec = Recommender(cfg, model.load(cfg), router, lc)
    sc = cfg["scoring"]

    log.info("Scoring %d pairs at hours %s%s", len(pairs), EVAL_HOURS,
             " (offline: cached pairs only)" if args.offline else "")
    rows, sens, failed, quota_hit = [], [], 0, None
    for _, p in pairs.iterrows():
        try:
            routes = cached_candidates(router, p, cache, args.sleep, args.offline)
        except QuotaExceeded as e:
            # keep whatever has been scored; the cache makes a later resume cheap
            quota_hit = int(p["pair"])
            log.warning("\nORS quota exhausted at pair %d: %s", quota_hit, e)
            log.warning("Re-run the same command later; cached pairs are not re-fetched.")
            break
        if not routes or len(routes) < 2:
            failed += 1
            continue
        for hour in EVAL_HOURS:
            stats = [rec._stats(r, hour, use_model=True) for r in routes]

            base = decide([s for s in map(_copy, stats)], sc["weights"],
                          sc["max_detour"], sc["min_cooling_c"])
            short, best = base.shortest, base.chosen
            rows.append({
                "pair": int(p["pair"]), "stratum": p.get("stratum", "all"),
                "mean_canopy_100m": p.get("mean_canopy_100m", float("nan")),
                "hour": hour, "n_candidates": len(routes),
                "shortest_m": round(short.distance_m), "chosen_m": round(best.distance_m),
                "detour_pct": round(100 * (best.distance_m / short.distance_m - 1), 2),
                "mean_drop_c": round(short.mean_lst_c - best.mean_lst_c, 3),
                "peak_drop_c": round(short.peak_lst_c - best.peak_lst_c, 3),
                "shade_gain_pct": round(best.shade_pct - short.shade_pct, 3),
                "changed": best.name != short.name,
                # what the best *eligible* candidate offered, before the min-cooling rule
                "best_available_drop_c": round(
                    short.mean_lst_c - min(s.mean_lst_c for s in stats if s.eligible), 3),
            })

            for detour in (1.1, 1.2, 1.3, 1.5):
                for wname, w in WEIGHT_SETS.items():
                    for min_cool in MIN_COOLING:
                        d2 = decide([s for s in map(_copy, stats)], w, detour, min_cool)
                        sens.append({"pair": int(p["pair"]), "hour": hour, "max_detour": detour,
                                     "weights": wname, "min_cooling_c": min_cool,
                                     "mean_drop_c": d2.shortest.mean_lst_c - d2.chosen.mean_lst_c,
                                     "detour_pct": 100 * (d2.chosen.distance_m / d2.shortest.distance_m - 1),
                                     "changed": d2.chosen.name != d2.shortest.name})
        if len(rows) % 60 == 0:
            log.info("  %d pair-hours scored", len(rows))
    lc.close()

    ev = pd.DataFrame(rows)
    ev.to_csv(tables / "route_eval.csv", index=False)
    if ev.empty:
        log.warning("No pair-hours scored this run (quota likely blocked on the very first "
                   "pair). Nothing to summarise; re-run once the quota resets.")
        return
    sn = pd.DataFrame(sens)
    agg = dict(mean_cooling_c=("mean_drop_c", "mean"),
               mean_detour_pct=("detour_pct", "mean"),
               changed_share=("changed", "mean"))
    # headline sweep: weights x detour cap, at the configured cooling threshold
    summary = (sn[sn["min_cooling_c"] == sc["min_cooling_c"]]
               .groupby(["weights", "max_detour"]).agg(**agg).round(3).reset_index())
    summary.to_csv(tables / "route_sensitivity.csv", index=False)
    # how much the minimum-cooling rule itself suppresses, at the default weights
    thr = (sn[sn["weights"] == DEFAULT_WEIGHTS]
           .groupby(["min_cooling_c", "max_detour"]).agg(**agg).round(3).reset_index())
    thr.to_csv(tables / "route_threshold.csv", index=False)

    log.info("\n%d pairs unroutable / skipped", failed)
    if quota_hit is not None:
        log.info("stopped early at pair %d (ORS quota); %d pairs scored so far",
                 quota_hit, ev["pair"].nunique())
    log.info("pair-hours scored: %d  (routes changed in %.0f%%)", len(ev), 100 * ev["changed"].mean())
    print("\n--- by hour ---")
    print(ev.groupby("hour")[["mean_drop_c", "peak_drop_c", "shade_gain_pct",
                              "detour_pct", "changed"]].mean().round(3).to_string())
    if ev["stratum"].nunique() > 1:
        strat = (ev.groupby("stratum")
                   .agg(n=("pair", "count"), canopy=("mean_canopy_100m", "mean"),
                        available=("best_available_drop_c", "mean"),
                        mean_drop_c=("mean_drop_c", "mean"),
                        detour_pct=("detour_pct", "mean"), changed=("changed", "mean"))
                   .round(3).reset_index())
        strat.to_csv(tables / "route_by_stratum.csv", index=False)
        print("\n--- by canopy stratum ---")
        print(strat.to_string(index=False))
    print("\n--- sensitivity (weights x detour cap) ---")
    print(summary.to_string(index=False))
    print("\n--- effect of the minimum-cooling rule (default weights) ---")
    print(thr.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for wname, g in summary.groupby("weights"):
        g = g.sort_values("mean_detour_pct")
        ax.plot(g["mean_detour_pct"], g["mean_cooling_c"], "o-", label=wname, linewidth=2, markersize=6)
        for _, r in g.iterrows():
            ax.annotate(f"{r['max_detour']:g}x", (r["mean_detour_pct"], r["mean_cooling_c"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7)
    ax.set_xlabel("mean detour over shortest route (%)")
    ax.set_ylabel("mean cooling vs shortest route (°C)")
    ax.set_title("Cooling bought per unit of detour")
    ax.grid(alpha=0.3)
    ax.legend(title="weights", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "tradeoff.png", dpi=160)
    print(f"\nTables -> {tables}\nFigure -> {figures / 'tradeoff.png'}")


MIN_COOLING = (0.0, 0.25, 0.5, 1.0)   # the "only detour if it buys something" threshold
DEFAULT_WEIGHTS = "heat-led (0.4/0.3/0.3)"
WEIGHT_SETS = {
    DEFAULT_WEIGHTS: {"heat": 0.4, "no_shade": 0.3, "distance": 0.3},
    "heat-only (1/0/0)": {"heat": 1.0, "no_shade": 0.0, "distance": 0.0},
    "shade-led (0.2/0.6/0.2)": {"heat": 0.2, "no_shade": 0.6, "distance": 0.2},
    "distance-led (0.2/0.2/0.6)": {"heat": 0.2, "no_shade": 0.2, "distance": 0.6},
}


def _copy(s):
    """decide() mutates eligible/score, so each setting is scored on a fresh copy."""
    from dataclasses import replace
    return replace(s)


if __name__ == "__main__":
    main()
