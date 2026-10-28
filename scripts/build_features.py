"""Task 3: pre-compute the derived rasters (neighbourhood land cover, coast distance).

    python -m scripts.build_features

Run once after scripts.clip_rasters and before scripts.build_dataset.
"""
import argparse
import logging

from shadewalk.config import load_config
from shadewalk.features import build_all


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = load_config(args.config)
    print("Building derived rasters...")
    for k, v in build_all(cfg).items():
        print(f"  {k:22} {v}")


if __name__ == "__main__":
    main()
