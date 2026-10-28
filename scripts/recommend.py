"""Step 3: recommend a walking route for the current time of day (system clock, LA time).

    python -m scripts.recommend --start "34.0224,-118.2851" --end "34.0522,-118.2437"
    python -m scripts.recommend --start "USC Village" --end "Grand Park Los Angeles"

Points are "lat,lon" or a place name (geocoded through openrouteservice).
"""
import argparse
import re

from shadewalk import model
from shadewalk.config import load_config
from shadewalk.landcover import LandCover
from shadewalk.recommend import Recommender
from shadewalk.routing import ORSClient

LATLON = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_point(text: str, router: ORSClient, cfg) -> tuple[float, float]:
    m = LATLON.match(text)
    if m:
        lat, lon = float(m[1]), float(m[2])
        return lon, lat
    return router.geocode(text, cfg.bbox)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    router = ORSClient(cfg)
    lc = LandCover.from_config(cfg)
    try:
        rec = Recommender(cfg, model.load(cfg), router, lc).recommend(
            parse_point(args.start, router, cfg), parse_point(args.end, router, cfg)
        )
        print(rec.summary())
    finally:
        lc.close()


if __name__ == "__main__":
    main()
