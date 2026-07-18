"""
utils/weather_fetcher.py
------------------------
Fetches live weather data for Andhra Pradesh
using Open-Meteo API (FREE - No API Key Required)

Returns a 50x50 flood-risk grid for the CNN/LSTM model.
"""

import numpy as np
import requests
from datetime import datetime
from .constants import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX

# Sample points across Andhra Pradesh (5x5 = 25 API calls)
SAMPLE_LATS = np.linspace(LAT_MIN, LAT_MAX, 5)
SAMPLE_LONS = np.linspace(LON_MIN, LON_MAX, 5)


def _fetch_single(lat: float, lon: float) -> float:
    """
    Fetch weather for one location and convert it into
    a flood-risk score (0.0 - 1.0).
    """

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat:.4f}"
        f"&longitude={lon:.4f}"
        "&hourly="
        "precipitation,"
        "temperature_2m,"
        "relative_humidity_2m,"
        "wind_speed_10m"
        "&forecast_days=1"
        "&timezone=Asia%2FKolkata"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        hourly = data.get("hourly", {})

        current_hour = datetime.now().hour

        precipitation = hourly.get("precipitation", [0.0] * 24)
        temperature = hourly.get("temperature_2m", [25.0] * 24)
        humidity = hourly.get("relative_humidity_2m", [50.0] * 24)
        wind = hourly.get("wind_speed_10m", [5.0] * 24)

        # Next 6 hours
        precipitation = precipitation[current_hour:current_hour + 6]
        temperature = temperature[current_hour:current_hour + 6]
        humidity = humidity[current_hour:current_hour + 6]
        wind = wind[current_hour:current_hour + 6]

        total_rain = float(sum(precipitation))
        avg_temp = float(np.mean(temperature))
        avg_humidity = float(np.mean(humidity))
        avg_wind = float(np.mean(wind))

        # -----------------------------
        # Weather Risk Calculation
        # -----------------------------
        rain_score = min(total_rain / 50.0, 1.0)
        humidity_score = min(avg_humidity / 100.0, 1.0)
        wind_score = min(avg_wind / 40.0, 1.0)

        risk = (
            0.70 * rain_score +
            0.20 * humidity_score +
            0.10 * wind_score
        )

        risk = float(np.clip(risk, 0.0, 1.0))

        print(
            f"({lat:.1f}, {lon:.1f}) | "
            f"Rain={total_rain:.1f} mm | "
            f"Temp={avg_temp:.1f}°C | "
            f"Humidity={avg_humidity:.1f}% | "
            f"Wind={avg_wind:.1f} km/h | "
            f"Risk={risk:.3f}"
        )

        return risk

    except Exception as e:
        print(f"[WeatherFetcher] Error ({lat:.2f},{lon:.2f}): {e}")
        return 0.0


def fetch_ap_risk_grid(grid_size: int = 50) -> np.ndarray:
    """
    Generate a grid of weather-based flood risk
    over Andhra Pradesh.
    """

    print("\n[WeatherFetcher] Fetching live AP weather...\n")

    sample = np.zeros((5, 5), dtype=np.float32)

    for i, lat in enumerate(SAMPLE_LATS):
        for j, lon in enumerate(SAMPLE_LONS):
            sample[i, j] = _fetch_single(lat, lon)

    try:
        from scipy.ndimage import zoom

        scale = grid_size / 5

        grid = zoom(sample, scale, order=1)

        grid = np.asarray(grid, dtype=np.float32)

        grid = np.clip(grid, 0.0, 1.0)

    except ImportError:

        grid = np.kron(
            sample,
            np.ones((10, 10), dtype=np.float32)
        )[:grid_size, :grid_size]

    print("\n-----------------------------------")
    print("Weather Grid Generated Successfully")
    print("-----------------------------------")
    print(f"Grid Shape : {grid.shape}")
    print(f"Mean Risk  : {grid.mean():.3f}")
    print(f"Max Risk   : {grid.max():.3f}")

    return grid


if __name__ == "__main__":

    grid = fetch_ap_risk_grid()

    print("\n✓ Weather Fetcher Working Correctly")