"""
utils/sentinel_preprocessing.py
Shared Sentinel-1 preprocessing for both historical training data and live inference.

Model input channels:
  0: VV backscatter normalized from [-25, 0] dB to [0, 1]
  1: VH backscatter normalized from [-32, -5] dB to [0, 1]
  2: VV - VH difference normalized from [0, 20] dB to [0, 1]

Output patch:
  (64, 64, 3), float32
"""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import ee
import numpy as np
import requests


PATCH_SIZE = 64
PIXEL_SCALE_M = 30
PATCH_SPAN_M = PATCH_SIZE * PIXEL_SCALE_M
INPUT_BANDS = ("VV_NORM", "VH_NORM", "VV_MINUS_VH")
DOWNLOAD_BANDS = INPUT_BANDS + ("VALID",)


@dataclass(frozen=True)
class PatchResult:
    patch: np.ndarray
    valid_fraction: float
    sha256: str


def initialize_ee(project: str) -> None:
    """Initialize Earth Engine, authenticating only when credentials are missing."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def sentinel1_collection(
    region: ee.Geometry,
    start_date: str,
    end_date: str,
    orbit: Optional[str] = None,
) -> ee.ImageCollection:
    """Return a homogeneous Sentinel-1 IW, VV+VH, 10 m collection."""
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
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
    )
    if orbit:
        collection = collection.filter(
            ee.Filter.eq("orbitProperties_pass", orbit)
        )
    return collection


def choose_orbit(
    region: ee.Geometry,
    start_date: str,
    end_date: str,
) -> Tuple[str, int]:
    """Choose the orbit pass with the largest number of scenes over a patch."""
    base = sentinel1_collection(region, start_date, end_date)
    counts = ee.Dictionary(
        {
            "ASCENDING": base.filter(
                ee.Filter.eq("orbitProperties_pass", "ASCENDING")
            ).size(),
            "DESCENDING": base.filter(
                ee.Filter.eq("orbitProperties_pass", "DESCENDING")
            ).size(),
        }
    ).getInfo()

    ascending = int(counts.get("ASCENDING", 0))
    descending = int(counts.get("DESCENDING", 0))

    if ascending <= 0 and descending <= 0:
        raise RuntimeError(
            f"No Sentinel-1 VV/VH IW scenes found from "
            f"{start_date} to {end_date}"
        )

    if ascending >= descending:
        return "ASCENDING", ascending
    return "DESCENDING", descending


def make_patch_region(lon: float, lat: float) -> ee.Geometry:
    """Create the square geographic region used for one 64x64 patch."""
    return (
        ee.Geometry.Point([float(lon), float(lat)])
        .buffer(PATCH_SPAN_M / 2.0)
        .bounds()
    )


def build_model_input(
    region: ee.Geometry,
    start_date: str,
    end_date: str,
    orbit: str,
) -> ee.Image:
    """Build the exact 3-channel image used by training and live inference."""
    collection = sentinel1_collection(
        region=region,
        start_date=start_date,
        end_date=end_date,
        orbit=orbit,
    )

    # Temporal median reduces outliers and part of the SAR speckle.
    composite = collection.select(["VV", "VH"]).median()

    # Small spatial median filter for additional speckle suppression.
    filtered = composite.focalMedian(30, "circle", "meters", 1)

    vv = filtered.select("VV")
    vh = filtered.select("VH")

    valid = (
        vv.mask()
        .And(vh.mask())
        .reduce(ee.Reducer.min())
        .rename("VALID")
        .toFloat()
    )

    vv_norm = (
        vv.clamp(-25.0, 0.0)
        .add(25.0)
        .divide(25.0)
        .rename("VV_NORM")
    )
    vh_norm = (
        vh.clamp(-32.0, -5.0)
        .add(32.0)
        .divide(27.0)
        .rename("VH_NORM")
    )
    vv_minus_vh = (
        vv.subtract(vh)
        .clamp(0.0, 20.0)
        .divide(20.0)
        .rename("VV_MINUS_VH")
    )

    return (
        ee.Image.cat([vv_norm, vh_norm, vv_minus_vh, valid])
        .unmask(0.0)
        .toFloat()
    )


def _structured_npy_to_array(data: np.ndarray) -> np.ndarray:
    if not data.dtype.names:
        raise ValueError("Earth Engine NPY response is not a structured array")

    missing = [name for name in DOWNLOAD_BANDS if name not in data.dtype.names]
    if missing:
        raise ValueError(f"Missing bands in downloaded patch: {missing}")

    return np.stack([data[name] for name in DOWNLOAD_BANDS], axis=-1)


def download_patch(
    image: ee.Image,
    region: ee.Geometry,
    session: Optional[requests.Session] = None,
    retries: int = 5,
    timeout_seconds: int = 120,
) -> PatchResult:
    """
    Download a small image directly as NPY.

    The VALID channel is used for quality control and is not returned to the CNN.
    """
    session = session or requests.Session()
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            url = image.getDownloadURL(
                {
                    "bands": list(DOWNLOAD_BANDS),
                    "region": region,
                    "dimensions": [PATCH_SIZE, PATCH_SIZE],
                    "format": "NPY",
                }
            )
            response = session.get(url, timeout=timeout_seconds)
            response.raise_for_status()

            structured = np.load(io.BytesIO(response.content), allow_pickle=False)
            array = _structured_npy_to_array(structured).astype(np.float32)

            if array.shape != (PATCH_SIZE, PATCH_SIZE, len(DOWNLOAD_BANDS)):
                raise ValueError(f"Unexpected patch shape: {array.shape}")

            valid_fraction = float(np.mean(array[..., 3] > 0.5))
            patch = np.clip(array[..., :3], 0.0, 1.0).astype(np.float32)

            if not np.isfinite(patch).all():
                raise ValueError("Patch contains NaN or infinite values")

            digest = hashlib.sha256(patch.tobytes()).hexdigest()
            return PatchResult(
                patch=patch,
                valid_fraction=valid_fraction,
                sha256=digest,
            )

        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(60, 2 ** attempt))

    raise RuntimeError(f"Patch download failed after {retries} attempts") from last_error
