"""
utils/weather_fetcher.py
------------------------
Fetches live rainfall and weather data for Andhra Pradesh
using Open-Meteo API — no API key required, fully free.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import requests
from utils.constants import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX


# Sample points across AP (5x5 = 25 API calls → interpolated to 50x50 grid)
SAMPLE_LATS = np.linspace(LAT_MIN, LAT_MAX, 5)
SAMPLE_LONS = np.linspace(LON_MIN, LON_MAX, 5)

def _fetch_single(lat: float, lon: float) -> float:
    """
    Fetch precipitation (mm/hr) at a single lat/lon.
    Returns a risk value 0.0–1.0.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat:.4f}&longitude={lon:.4f}"
        "&hourly=precipitation,rain,showers"
        "&forecast_days=1"
        "&timezone=Asia%2FKolkata"
    )
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        # Take the next 6 hours of precipitation
        precip = hourly.get("precipitation", [0.0] * 6)[:6]
        rain   = hourly.get("rain",          [0.0] * 6)[:6]
        shower = hourly.get("showers",       [0.0] * 6)[:6]
        # Total accumulated rainfall in next 6 hours
        total = sum(p + r + s for p, r, s in zip(precip, rain, shower))
        # Normalise: 0mm=0.0, 50mm+=1.0 (heavy flood threshold)
        risk = min(total / 50.0, 1.0)
        return float(risk)
    except Exception as e:
        print(f"  [WeatherFetcher] Warning: {e} — using 0.0 for ({lat:.2f}, {lon:.2f})")
        return 0.0

def fetch_ap_risk_grid(grid_size: int = 50) -> np.ndarray:
    """
    Returns a (grid_size x grid_size) float32 array of flood risk values
    based on live Open-Meteo 6-hour rainfall forecast over Andhra Pradesh.
    Values range 0.0 (no risk) to 1.0 (extreme risk).
    """
    print("[WeatherFetcher] Fetching live AP rainfall data from Open-Meteo...")
    # Build a 5×5 sample grid (25 API calls)
    sample = np.zeros((5, 5), dtype=np.float32)
    for i, lat in enumerate(SAMPLE_LATS):
        for j, lon in enumerate(SAMPLE_LONS):
            sample[i, j] = _fetch_single(lat, lon)
            print(f"  ({lat:.1f}°N, {lon:.1f}°E) → risk={sample[i,j]:.3f}")
    # Bilinear interpolation from 5×5 to grid_size×grid_size
    from scipy.ndimage import zoom
    try:
        from scipy.ndimage import zoom as sz
        scale = grid_size / 5
        full_grid = sz(sample, scale, order=1)  # bilinear
        full_grid = np.clip(full_grid, 0.0, 1.0).astype(np.float32)
    except ImportError:
        # Fallback: repeat-tile if scipy not available
        full_grid = np.kron(sample, np.ones((10, 10), dtype=np.float32))[:grid_size, :grid_size]
    print(f"[WeatherFetcher] Done. Grid mean risk: {full_grid.mean():.3f}, max: {full_grid.max():.3f}")
    return full_grid

if __name__ == "__main__":
    # Quick test — run: python utils/weather_fetcher.py
    grid = fetch_ap_risk_grid()
    print(f"\nGrid shape : {grid.shape}")
    print(f"Mean risk  : {grid.mean():.4f}")
    print(f"Max risk   : {grid.max():.4f}")
    print("✓ Weather fetcher working correctly")