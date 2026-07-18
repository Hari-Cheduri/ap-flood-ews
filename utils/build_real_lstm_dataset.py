from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from sklearn.preprocessing import RobustScaler
from urllib3.util.retry import Retry

from utils.lstm_weather_contract import (
    CONTRACT_NAME, DATASET_SUMMARY, FEATURES, FORECAST_HORIZON,
    INPUT_SHAPE, SCALER_PATH, SEQUENCE_LENGTH,
    X_TEST, X_TRAIN, X_VAL, Y_TEST, Y_TRAIN, Y_VAL,
)

DISTRICTS: Dict[str, Tuple[float, float]] = {
    "alluri_sitharama_raju": (18.08, 82.66),
    "anakapalli": (17.69, 83.00),
    "ananthapuramu": (14.68, 77.60),
    "annamayya": (14.05, 78.75),
    "bapatla": (15.90, 80.47),
    "chittoor": (13.22, 79.10),
    "east_godavari": (16.99, 81.78),
    "eluru": (16.71, 81.10),
    "guntur": (16.31, 80.44),
    "kakinada": (16.99, 82.25),
    "konaseema": (16.58, 82.01),
    "krishna": (16.18, 81.13),
    "kurnool": (15.83, 78.04),
    "nandyal": (15.48, 78.48),
    "ntr_vijayawada": (16.51, 80.65),
    "palnadu": (16.24, 80.05),
    "parvathipuram_manyam": (18.78, 83.43),
    "prakasam": (15.50, 80.05),
    "spsr_nellore": (14.44, 79.99),
    "sri_sathya_sai": (14.17, 77.81),
    "srikakulam": (18.29, 83.90),
    "tirupati": (13.63, 79.42),
    "visakhapatnam": (17.69, 83.22),
    "vizianagaram": (18.10, 83.40),
    "west_godavari": (16.54, 81.52),
    "ysr_kadapa": (14.47, 78.82),
}
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--stride-hours", type=int, default=6)
    parser.add_argument("--negative-keep-prob", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-districts", type=int, default=None)
    return parser.parse_args()


def make_session() -> requests.Session:
    retry = Retry(
        total=5, connect=5, read=5, status=5, backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def yearly_ranges(start_text: str, end_text: str) -> Iterable[Tuple[str, str]]:
    start = date.fromisoformat(start_text)
    end = date.fromisoformat(end_text)
    if end < start:
        raise ValueError("end date must not be before start date")
    current = start
    while current <= end:
        year_end = min(date(current.year, 12, 31), end)
        yield current.isoformat(), year_end.isoformat()
        current = year_end + timedelta(days=1)


def fetch_hourly(
    session: requests.Session,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> Tuple[List[str], np.ndarray]:
    response = session.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(FEATURES),
            "timezone": "Asia/Kolkata",
        },
        timeout=120,
    )
    response.raise_for_status()
    hourly = (response.json().get("hourly") or {})
    times = hourly.get("time") or []
    if not times:
        raise RuntimeError("Open-Meteo returned no hourly timestamps")

    columns = []
    for feature in FEATURES:
        values = hourly.get(feature)
        if values is None or len(values) != len(times):
            raise RuntimeError(f"Missing hourly feature: {feature}")
        columns.append(np.asarray(
            [np.nan if value is None else float(value) for value in values],
            dtype=np.float32,
        ))

    matrix = np.column_stack(columns).astype(np.float32)
    positions = np.arange(len(matrix))
    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        finite = np.isfinite(column)
        if not finite.any():
            raise RuntimeError(f"No valid values for {FEATURES[column_index]}")
        if (~finite).any():
            column[~finite] = np.interp(
                positions[~finite], positions[finite], column[finite]
            )
            matrix[:, column_index] = column
    return times, matrix


def make_label(sequence: np.ndarray, future: np.ndarray) -> int:
    rain_i = FEATURES.index("precipitation")
    humidity_i = FEATURES.index("relative_humidity_2m")

    past72 = float(sequence[:, rain_i].sum())
    past24 = float(sequence[-24:, rain_i].sum())
    future24 = float(future[:, rain_i].sum())
    peak_hour = float(future[:, rain_i].max())
    peak6 = max(
        float(future[start:start + 6, rain_i].sum())
        for start in range(0, len(future) - 5)
    )
    humidity24 = float(sequence[-24:, humidity_i].mean())

    high_risk = (
        future24 >= 35.0
        or peak_hour >= 8.0
        or peak6 >= 20.0
        or (past72 >= 60.0 and future24 >= 20.0)
        or (past24 >= 35.0 and humidity24 >= 88.0 and future24 >= 15.0)
    )
    return int(high_risk)


def split_name(timestamp: np.datetime64) -> str:
    if timestamp < np.datetime64("2024-07-01T00:00"):
        return "train"
    if timestamp < np.datetime64("2025-04-01T00:00"):
        return "val"
    return "test"


def distribution(y: np.ndarray) -> Dict[str, int]:
    return {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))}


def main() -> None:
    args = parse_args()
    if args.stride_hours < 1:
        raise SystemExit("--stride-hours must be positive")
    if not (0.0 < args.negative_keep_prob <= 1.0):
        raise SystemExit("--negative-keep-prob must be in (0,1]")

    rng = np.random.default_rng(args.seed)
    session = make_session()
    districts = list(DISTRICTS.items())
    if args.max_districts is not None:
        districts = districts[:args.max_districts]

    split_X = {name: [] for name in ("train", "val", "test")}
    split_y = {name: [] for name in ("train", "val", "test")}
    failures = []

    print("\n[LSTM Dataset] Building real hourly weather sequences")
    print("Districts:", len(districts))
    print("Period   :", args.start_date, "to", args.end_date)
    print("Input    :", INPUT_SHAPE)
    print("Target   : next 24-hour heavy-rain risk proxy")

    for number, (district, (lat, lon)) in enumerate(districts, 1):
        print(f"\n[{number}/{len(districts)}] {district}")
        all_times, blocks = [], []

        for chunk_start, chunk_end in yearly_ranges(
            args.start_date, args.end_date
        ):
            try:
                print("  Fetching", chunk_start, "to", chunk_end)
                times, matrix = fetch_hourly(
                    session, lat, lon, chunk_start, chunk_end
                )
                all_times.extend(times)
                blocks.append(matrix)
                time.sleep(0.15)
            except Exception as exc:
                message = f"{district} {chunk_start}..{chunk_end}: {exc}"
                failures.append(message)
                print("  [ERROR]", message)

        if not blocks:
            continue

        matrix = np.vstack(blocks)
        timestamps = np.asarray(all_times, dtype="datetime64[m]")
        order = np.argsort(timestamps)
        timestamps, matrix = timestamps[order], matrix[order]
        timestamps, unique_indices = np.unique(timestamps, return_index=True)
        matrix = matrix[unique_indices]

        accepted = {0: 0, 1: 0}
        for endpoint in range(
            SEQUENCE_LENGTH,
            len(matrix) - FORECAST_HORIZON,
            args.stride_hours,
        ):
            sequence = matrix[endpoint - SEQUENCE_LENGTH:endpoint]
            future = matrix[endpoint:endpoint + FORECAST_HORIZON]
            label = make_label(sequence, future)

            if label == 0 and rng.random() > args.negative_keep_prob:
                continue

            split = split_name(timestamps[endpoint])
            split_X[split].append(sequence.astype(np.float32))
            split_y[split].append(label)
            accepted[label] += 1

        print("  Accepted:", accepted)

    arrays = {}
    for split in ("train", "val", "test"):
        if not split_X[split]:
            raise RuntimeError(f"No {split} sequences were created")
        X = np.stack(split_X[split]).astype(np.float32)
        y = np.asarray(split_y[split], dtype=np.int8)
        if len(np.unique(y)) < 2:
            raise RuntimeError(
                f"{split} has one class only. Expand dates or keep more negatives."
            )
        perm = rng.permutation(len(y))
        arrays[split] = (X[perm], y[perm])

    X_train_raw, y_train = arrays["train"]
    X_val_raw, y_val = arrays["val"]
    X_test_raw, y_test = arrays["test"]

    scaler = RobustScaler()
    scaler.fit(X_train_raw.reshape(-1, len(FEATURES)))

    def scale(X: np.ndarray) -> np.ndarray:
        return scaler.transform(
            X.reshape(-1, len(FEATURES))
        ).reshape(X.shape).astype(np.float32)

    X_train, X_val, X_test = map(
        scale, (X_train_raw, X_val_raw, X_test_raw)
    )

    X_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(X_TRAIN, X_train, allow_pickle=False)
    np.save(X_VAL, X_val, allow_pickle=False)
    np.save(X_TEST, X_test, allow_pickle=False)
    np.save(Y_TRAIN, y_train, allow_pickle=False)
    np.save(Y_VAL, y_val, allow_pickle=False)
    np.save(Y_TEST, y_test, allow_pickle=False)
    joblib.dump(scaler, SCALER_PATH)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": CONTRACT_NAME,
        "source": "Open-Meteo Historical Weather API",
        "features": list(FEATURES),
        "input_shape": list(INPUT_SHAPE),
        "label": "Weak next-24-hour weather-driven flood-risk proxy",
        "splits": {
            "train": {"shape": list(X_train.shape), "classes": distribution(y_train)},
            "val": {"shape": list(X_val.shape), "classes": distribution(y_val)},
            "test": {"shape": list(X_test.shape), "classes": distribution(y_test)},
        },
        "request_failures": failures,
    }
    DATASET_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n========================================")
    print("LSTM DATASET COMPLETE")
    print("========================================")
    for name, (_, y) in arrays.items():
        print(name, distribution(y))
    print("Scaler :", SCALER_PATH)
    print("Summary:", DATASET_SUMMARY)


if __name__ == "__main__":
    main()
