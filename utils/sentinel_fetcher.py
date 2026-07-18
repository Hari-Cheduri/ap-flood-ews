"""
Fetch a live local Sentinel-1 patch using the same preprocessing contract as
the trained CNN, while recording actual acquisition timestamps and freshness.

Important
---------
The patch is a median composite when more than one scene exists in the selected
window. Freshness is based on the newest contributing scene.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import ee
import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.constants import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from utils.sentinel_cnn_contract import (
    BUFFER_KM,
    CONTRACT_NAME,
    IMAGE_SIZE,
    LIVE_METADATA_PATH,
    LIVE_PATCH_PATH,
    TRAINING_WINDOW_DAYS,
    VIS_MAX,
    VIS_MIN,
    validate_patch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a freshness-aware CNN-compatible Sentinel-1 patch."
    )
    parser.add_argument("--project", default="ap-flood-monitor")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--name", default="requested_location")
    parser.add_argument("--window-days", type=int, default=TRAINING_WINDOW_DAYS)
    parser.add_argument(
        "--max-lookback-days",
        type=int,
        default=72,
        help="Search older windows when the newest window has no scenes.",
    )
    parser.add_argument("--buffer-km", type=float, default=BUFFER_KM)
    parser.add_argument("--output", type=Path, default=LIVE_PATCH_PATH)
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=LIVE_METADATA_PATH,
    )
    return parser.parse_args()


def request_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def initialize_ee(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate first and "
            f"confirm project '{project}' is enabled."
        ) from exc


def ee_info(value: Any, description: str, retries: int = 4) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return value.getInfo()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            delay = min(20, 2 ** attempt)
            print(
                f"[Retry] {description} failed "
                f"({attempt}/{retries}): {exc}"
            )
            time.sleep(delay)
    raise RuntimeError(f"Failed while {description}") from last_error


def make_aoi(lat: float, lon: float, buffer_km: float) -> ee.Geometry:
    return (
        ee.Geometry.Point([float(lon), float(lat)])
        .buffer(float(buffer_km) * 1000.0)
        .bounds()
    )


def sentinel_collection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
) -> ee.ImageCollection:
    # Keep this aligned with the historical CNN dataset contract.
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .select(["VV", "VH"])
    )


def find_latest_window(
    aoi: ee.Geometry,
    window_days: int,
    max_lookback_days: int,
) -> Tuple[ee.ImageCollection, str, str, int]:
    if window_days <= 0:
        raise ValueError("--window-days must be positive")
    if max_lookback_days < window_days:
        raise ValueError("--max-lookback-days must be >= --window-days")

    latest_end = date.today() + timedelta(days=1)

    for offset in range(0, max_lookback_days, window_days):
        end = latest_end - timedelta(days=offset)
        start = end - timedelta(days=window_days)
        start_text = start.isoformat()
        end_text = end.isoformat()

        collection = sentinel_collection(aoi, start_text, end_text)
        count = int(
            ee_info(
                collection.size(),
                f"counting Sentinel-1 scenes for {start_text} to {end_text}",
            )
        )
        print(
            f"[Sentinel-1] Window {start_text} to {end_text}: "
            f"{count} scenes"
        )
        if count > 0:
            return collection, start_text, end_text, count

    raise RuntimeError(
        "No Sentinel-1 VV/VH IW scenes were found in the configured "
        f"{max_lookback_days}-day search period."
    )


def _utc_from_millis(value: int | float) -> datetime:
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def freshness_policy(age_days: float) -> Dict[str, Any]:
    """
    Project policy, not an official disaster-management standard.

    The weights are conservative and should later be calibrated with verified
    event data.
    """
    age_days = max(0.0, float(age_days))

    if age_days <= 3.0:
        label, factor, usable = "FRESH", 1.00, True
    elif age_days <= 6.0:
        label, factor, usable = "RECENT", 0.85, True
    elif age_days <= 12.0:
        label, factor, usable = "AGING", 0.60, True
    elif age_days <= 24.0:
        label, factor, usable = "STALE", 0.30, True
    else:
        label, factor, usable = "EXPIRED", 0.00, False

    return {
        "class": label,
        "age_days": round(age_days, 3),
        "factor": factor,
        "usable": usable,
        "policy": {
            "fresh_max_days": 3,
            "recent_max_days": 6,
            "aging_max_days": 12,
            "stale_max_days": 24,
            "expired_after_days": 24,
        },
    }


def collection_time_metadata(
    collection: ee.ImageCollection,
) -> Dict[str, Any]:
    raw_times = ee_info(
        collection.aggregate_array("system:time_start"),
        "reading Sentinel-1 acquisition timestamps",
    )
    if not isinstance(raw_times, list) or not raw_times:
        raise RuntimeError(
            "Sentinel-1 collection contains no system:time_start values"
        )

    datetimes = sorted(_utc_from_millis(value) for value in raw_times)
    now_utc = datetime.now(timezone.utc)
    latest = datetimes[-1]
    oldest = datetimes[0]
    age_days = (now_utc - latest).total_seconds() / 86400.0

    freshness = freshness_policy(age_days)
    freshness.update({
        "latest_scene_utc": latest.isoformat(),
        "oldest_scene_utc": oldest.isoformat(),
        "composite_span_days": round(
            (latest - oldest).total_seconds() / 86400.0,
            3,
        ),
    })

    return {
        "scene_times_utc": [item.isoformat() for item in datetimes],
        "latest_scene_utc": latest.isoformat(),
        "oldest_scene_utc": oldest.isoformat(),
        "satellite_freshness": freshness,
    }


def build_rgb_image(
    collection: ee.ImageCollection,
    aoi: ee.Geometry,
) -> ee.Image:
    median = collection.median().clip(aoi)
    vv = median.select("VV")
    vh = median.select("VH")
    difference = vv.subtract(vh).rename("VV_minus_VH")
    return vv.addBands(vh).addBands(difference)


def download_png_patch(
    image: ee.Image,
    aoi: ee.Geometry,
    size: int,
    session: requests.Session,
) -> np.ndarray:
    url = image.getThumbURL(
        {
            "region": aoi,
            "dimensions": f"{size}x{size}",
            "format": "png",
            "min": list(VIS_MIN),
            "max": list(VIS_MAX),
        }
    )
    response = session.get(url, timeout=90)
    response.raise_for_status()

    png = Image.open(BytesIO(response.content)).convert("RGB")
    patch = np.asarray(png, dtype=np.float32) / 255.0
    return validate_patch(patch, "live Sentinel-1 patch")


def main() -> None:
    args = parse_args()

    if not (float(LAT_MIN) <= args.lat <= float(LAT_MAX)):
        raise SystemExit(
            f"Latitude {args.lat} is outside configured bounds "
            f"[{LAT_MIN}, {LAT_MAX}]"
        )
    if not (float(LON_MIN) <= args.lon <= float(LON_MAX)):
        raise SystemExit(
            f"Longitude {args.lon} is outside configured bounds "
            f"[{LON_MIN}, {LON_MAX}]"
        )

    print(f"[GEE] Initializing project: {args.project}")
    initialize_ee(args.project)
    print("[GEE] Earth Engine initialized")

    aoi = make_aoi(args.lat, args.lon, args.buffer_km)
    collection, start_date, end_date, scene_count = find_latest_window(
        aoi=aoi,
        window_days=args.window_days,
        max_lookback_days=args.max_lookback_days,
    )
    temporal = collection_time_metadata(collection)

    image = build_rgb_image(collection, aoi)
    patch = download_png_patch(
        image=image,
        aoi=aoi,
        size=IMAGE_SIZE,
        session=request_session(),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, patch, allow_pickle=False)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": CONTRACT_NAME,
        "location_name": args.name,
        "latitude": float(args.lat),
        "longitude": float(args.lon),
        "buffer_km": float(args.buffer_km),
        "window_days": int(args.window_days),
        "search_window_start": start_date,
        "search_window_end_exclusive": end_date,
        "sentinel1_scene_count": int(scene_count),
        "composite_method": "median",
        **temporal,
        "shape": list(patch.shape),
        "dtype": str(patch.dtype),
        "range": {
            "min": float(patch.min()),
            "max": float(patch.max()),
            "mean": float(patch.mean()),
            "std": float(patch.std()),
        },
        "bands": [
            "VV rendered [-25, 0] dB",
            "VH rendered [-30, -5] dB",
            "VV minus VH rendered [0, 20] dB",
        ],
    }

    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    freshness = metadata["satellite_freshness"]
    print("\n[Sentinel-1] CNN-compatible patch saved")
    print(f"Patch       : {args.output}")
    print(f"Metadata    : {args.metadata_output}")
    print(f"Contract    : {CONTRACT_NAME}")
    print(f"Location    : {args.name} ({args.lat}, {args.lon})")
    print(f"Window      : {start_date} to {end_date}")
    print(f"Scenes      : {scene_count} ({metadata['composite_method']} composite)")
    print(f"Latest scene: {metadata['latest_scene_utc']}")
    print(
        "Freshness   :",
        freshness["class"],
        f"age={freshness['age_days']:.2f} days",
        f"factor={freshness['factor']:.2f}",
    )
    print(f"Shape       : {patch.shape}")
    print(f"Range       : {patch.min():.4f} to {patch.max():.4f}")
    print(f"Mean/std    : {patch.mean():.4f} / {patch.std():.4f}")


if __name__ == "__main__":
    main()
