"""End-to-end test on small synthetic rasters with a fake openrouteservice.

Files mimic the real layouts: ECOSTRESS tiles in UTM 11N (EPSG:32611, 70 m, Kelvin),
NLCD rasters in CONUS Albers (EPSG:5070, 30 m).
"""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import rasterio
import yaml
from pyproj import Transformer
from rasterio.transform import from_origin

from shadewalk import model
from shadewalk.config import PROJECT_ROOT, load_config
from shadewalk.dataset import build_dataset
from shadewalk.landcover import LandCover
from shadewalk.recommend import Recommender
from shadewalk.routing import ORSClient

LA = ZoneInfo("America/Los_Angeles")
PARK = (-118.27, 34.03)  # shaded strip in the synthetic world


def canopy_at(lon, lat):
    park = np.exp(-(((lon - PARK[0]) / 0.006) ** 2))  # north-south green corridor
    return np.clip(10 + 70 * park, 0, 100)


def impervious_at(lon, lat):
    return np.clip(90 - 0.9 * canopy_at(lon, lat), 0, 100)


def lst_c_at(lon, lat, hour):
    di = np.sin(np.pi * (hour - 6) / 16)
    can, imp = canopy_at(lon, lat), impervious_at(lon, lat)
    return 25 + 0.08 * imp - 0.12 * can + (4 + 8 * imp / 100) * di


def write(path, arr, crs, transform, nodata=None):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs=crs, transform=transform, nodata=nodata) as dst:
        dst.write(arr, 1)


def grid_lonlat(crs, transform, shape):
    rows, cols = np.indices(shape)
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel(), offset="center")
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(np.asarray(xs), np.asarray(ys))
    return lon.reshape(shape), lat.reshape(shape)


@pytest.fixture()
def project(tmp_path):
    rng = np.random.default_rng(0)
    eco, nlcd = tmp_path / "eco", tmp_path / "nlcd"
    eco.mkdir(), nlcd.mkdir()

    # NLCD rasters (30 m, EPSG:5070) covering the test area
    a_x, a_y = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform(-118.33, 34.07)
    t5070 = from_origin(a_x, a_y, 30, 30)
    lon, lat = grid_lonlat("EPSG:5070", t5070, (300, 300))
    write(nlcd / "canopy.tif", canopy_at(lon, lat).astype("uint8"), "EPSG:5070", t5070, nodata=255)
    write(nlcd / "impervious.tif", impervious_at(lon, lat).astype("uint8"), "EPSG:5070", t5070, nodata=127)

    # ECOSTRESS granules (70 m, EPSG:32611). Local hours 9..17 plus one night pass.
    u_x, u_y = Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True).transform(-118.32, 34.065)
    t326 = from_origin(u_x, u_y, 70, 70)
    shape = (110, 110)
    lon, lat = grid_lonlat("EPSG:32611", t326, shape)
    local_hours = [9, 10, 11, 12, 13, 14, 15, 16, 17, 0.5]
    for k, h in enumerate(local_hours):
        local = datetime(2026, 7, 10 + k, int(h), int((h % 1) * 60), tzinfo=LA)
        ts = local.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = f"ECOv002_L2T_LSTE_{44800 + k}_00{k}_11SLT_{ts}_0713_01"
        lst = (lst_c_at(lon, lat, h) + rng.normal(0, 0.7, shape) + 273.15).astype("float32")
        cloud = (rng.random(shape) < 0.1).astype("uint8")
        qc = np.where(rng.random(shape) < 0.05, 2, 0).astype("uint16")  # 2 = not usable
        write(eco / f"{key}_LST.tif", lst, "EPSG:32611", t326, nodata=np.nan)
        write(eco / f"{key}_cloud.tif", cloud, "EPSG:32611", t326)
        write(eco / f"{key}_QC.tif", qc, "EPSG:32611", t326)
        write(eco / f"{key}_water.tif", np.zeros(shape, "uint8"), "EPSG:32611", t326)
        write(eco / f"{key}_height.tif", np.zeros(shape, "float32"), "EPSG:32611", t326)  # ignored layer

    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    cfg["data"] = {"ecostress_dir": str(eco), "canopy_tif": str(nlcd / "canopy.tif"),
                   "impervious_tif": str(nlcd / "impervious.tif"), "dataset_csv": str(tmp_path / "train.csv")}
    # the synthetic world has no derived rasters, so train on the base feature set
    cfg["features"] = {"radius_m": 100, "columns": ["lat", "lon", "canopy_pct", "impervious_pct", "hour"]}
    cfg["reports"] = {"tables_dir": str(tmp_path / "tables"), "figures_dir": str(tmp_path / "figures")}
    cfg["model"].update(path=str(tmp_path / "rf.joblib"), metrics_path=str(tmp_path / "m.json"),
                        n_estimators=60, block_size_deg=0.01, target="lst", eval_split="spatial",
                        n_folds=2, max_depth=None)
    cfg["preprocess"]["samples_per_granule"] = 1500
    cfg["routing"]["via_offsets"] = []
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    return load_config(tmp_path / "config.yaml")


class FakeResp:
    def __init__(self, payload): self.payload, self.status_code, self.text = payload, 200, ""
    def json(self): return self.payload
    def raise_for_status(self): pass


class FakeORS:
    """Two routes A->B: a straight hot one and a longer one through the green corridor."""
    def post(self, url, json, headers, timeout):
        (x1, y1), (x2, y2) = json["coordinates"][0], json["coordinates"][-1]
        hot = [[x1, y1], [x2, y2]]
        cool = [[x1, y1], [PARK[0], y1], [PARK[0], y2], [x2, y2]]
        feats = [{"geometry": {"coordinates": hot}, "properties": {"summary": {"distance": 1800.0}}},
                 {"geometry": {"coordinates": cool}, "properties": {"summary": {"distance": 2100.0}}}]
        assert "alternative_routes" in json
        return FakeResp({"features": feats})


def test_pipeline_end_to_end(project):
    df = build_dataset(project)
    assert df["hour"].between(8, 19).all(), "night granule must be filtered out"
    assert df["granule"].nunique() == 9
    assert df["lst_c"].between(15, 60).all(), "LST must be converted from Kelvin"

    rf, metrics = model.train(df, project)
    assert metrics["random_forest"]["r2_mean"] > 0.8
    assert metrics["random_forest"]["rmse_c_mean"] < metrics["linear_baseline"]["rmse_c_mean"]
    model.save(rf, metrics, project)

    lc = LandCover(project.path("data", "canopy_tif"), project.path("data", "impervious_tif"))
    rec = Recommender(project, model.load(project), ORSClient(project, api_key="test", session=FakeORS()), lc)
    start, end = (-118.30, 34.045), (-118.24, 34.015)

    afternoon = rec.recommend(start, end, now=datetime(2026, 8, 3, 14, 10, tzinfo=LA))
    assert afternoon.daytime
    assert afternoon.decision.chosen.name == "R2", afternoon.summary()
    assert afternoon.decision.chosen.shade_pct > afternoon.decision.shortest.shade_pct

    night = rec.recommend(start, end, now=datetime(2026, 8, 3, 23, 30, tzinfo=LA))
    assert not night.daytime
    assert night.decision.chosen.name == "R1"
    print("\n" + afternoon.summary() + "\n\n" + night.summary())
    lc.close()


def test_uses_system_clock(project, monkeypatch):
    """Without `now`, the recommender reads the current LA time."""
    import shadewalk.recommend as r
    fixed = datetime(2026, 8, 3, 13, 0, tzinfo=LA)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return fixed.astimezone(tz)

    monkeypatch.setattr(r, "datetime", Clock)
    df = build_dataset(project)
    rf, m = model.train(df, project)
    lc = LandCover(project.path("data", "canopy_tif"), project.path("data", "impervious_tif"))
    out = r.Recommender(project, rf, ORSClient(project, api_key="t", session=FakeORS()), lc).recommend(
        (-118.30, 34.045), (-118.24, 34.015))
    assert out.local_time.hour == 13 and out.daytime
    lc.close()
