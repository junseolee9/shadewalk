"""Step 2: train the Random Forest LST model and compare it with linear regression.

    python -m scripts.train
"""
import argparse
import json

import pandas as pd

from shadewalk import model
from shadewalk.config import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(cfg.path("data", "dataset_csv"))
    rf, metrics = model.train(df, cfg)
    model.save(rf, metrics, cfg)
    print(json.dumps(metrics, indent=2))
    print(f"Model saved -> {cfg.path('model', 'path')}")


if __name__ == "__main__":
    main()
