# ShadeWalk LA

Heat- and shade-aware walking routes for Los Angeles (CSCI-461 project).
At request time the system reads the **current LA time from the system clock** and
recommends the route for that hour. Outside the daytime window (08:00–19:00) it returns
the shortest route, since night-time safety scoring is future work.

## Pipeline

| Step | Code | What it does |
|---|---|---|
| 1. Data | `shadewalk/ecostress.py`, `landcover.py`, `dataset.py` | Daytime ECOSTRESS LST pixels (cloud/QC/water masked, K → °C) joined with NLCD canopy % and impervious % |
| 2. Model | `shadewalk/model.py` | Random Forest regression of LST from `lat, lon, canopy_pct, impervious_pct, hour`; linear-regression baseline; spatial block train/test split; RMSE / MAE / R² |
| 3. Routes | `shadewalk/routing.py` | openrouteservice `foot-walking` alternative routes (+ optional side-point routes), sampled every 50 m |
| 4. Rank | `shadewalk/scoring.py`, `recommend.py` | Detour filter (≤ 1.3× shortest), score = a·heat + b·(no shade) + c·distance, keep shortest unless ≥ 1 °C cooler |

All thresholds and weights live in `config.yaml`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # then paste your key into .env (free: https://openrouteservice.org/dev/#/signup)
```

## Data

Put files here (paths are set in `config.yaml`):

```
data/raw/ecostress/   all downloaded ECOv002_L2T_LSTE_*.tif files (subfolders are fine)
data/raw/nlcd/canopy.tif       USFS Tree Canopy Cover (NLCD), CONUS or clipped to LA
data/raw/nlcd/impervious.tif   NLCD Impervious Surface, CONUS or clipped to LA
```

- ECOSTRESS: Earthdata Search → `ECO_L2T_LSTE v002`, LA box, summer dates, Day filter.
  Only the `LST`, `QC`, `cloud`, `water` layers are read; others are ignored.
- NLCD canopy and impervious: https://www.mrlc.gov/data. Any CRS works; points are reprojected.
  Clipping to LA first makes sampling much faster.

## Run

```bash
# one-off data preparation
python -m scripts.clip_rasters       # CONUS impervious raster -> LA-only copy
python -m scripts.build_features     # neighbourhood means, coast distance, DEM mosaic

# main pipeline
python -m scripts.build_dataset      # -> data/processed/training.csv + reports/tables/data_funnel.csv
python -m scripts.train              # -> models/rf_lst.joblib, models/metrics.json

# experiments behind the report
python -m scripts.experiments        # model x target x split grid -> reports/tables/
python -m scripts.robustness         # LODO, nested CV, ablation, ranking check
python -m scripts.evaluate_routes --pairs 100 --sleep 8   # add --offline to re-score the cache
python -m scripts.make_figures       # -> reports/figures/

# ad-hoc recommendation
python -m scripts.recommend --start "34.0224,-118.2851" --end "34.0522,-118.2437"
python -m scripts.recommend --start "USC Village" --end "Grand Park Los Angeles"

python -m pytest -q                  # end-to-end test on synthetic rasters, no API key needed
```

## Things to verify with the real files

- **QC bits.** `keep_qc_values: [0, 1]` keeps bits 0–1 = best/nominal. Confirm against the ECOSTRESS L2 user guide.
- **LST units.** v002 tiles are expected as float32 Kelvin; integer-scaled files are handled with the 0.02 scale factor.
- **ORS alternatives.** Only allowed for 2-point requests; short trips may return just one route. The side-point routes in `via_offsets` add variety.
- **Hour range.** The model only sees the hours present in the downloaded granules; check the `hour` column in `training.csv`.
