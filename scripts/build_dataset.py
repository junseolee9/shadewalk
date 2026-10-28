"""Step 1: ECOSTRESS daytime pixels + NLCD land cover -> data/processed/training.csv

    python -m scripts.build_dataset
"""
import argparse
import logging

from shadewalk.config import load_config
from shadewalk.dataset import build_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    df = build_dataset(cfg)
    out = cfg.path("data", "dataset_csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows from {df['granule'].nunique()} granules -> {out}")
    print(df[["lst_c", "canopy_pct", "impervious_pct", "hour"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
