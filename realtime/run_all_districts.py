"""
Run the existing full CNN + LSTM pipeline for multiple AP districts.

Each single-location run uses the existing singleton live files. Immediately
after the run, this script archives those files into a district-specific folder
before the next district overwrites them.

Examples
--------
Run all 26 districts:
    python -m realtime.run_all_districts --project ap-flood-monitor

Run selected districts:
    python -m realtime.run_all_districts `
        --project ap-flood-monitor `
        --district ntr_vijayawada `
        --district guntur `
        --district krishna

Resume an interrupted all-district run:
    python -m realtime.run_all_districts `
        --project ap-flood-monitor `
        --resume
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from realtime.build_ap_risk_map_data import build_outputs
from utils.ap_districts import AP_DISTRICTS, DISTRICTS_BY_SLUG, District


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_ROOT = PROCESSED_DIR / "district_results"

FILES_TO_ARCHIVE = (
    "sentinel1_live_patch.npy",
    "sentinel1_live_patch.json",
    "live_cnn_prediction.json",
    "lstm_live_raw.npy",
    "lstm_live_sequence.npy",
    "lstm_live_sequence.json",
    "live_lstm_prediction.json",
    "hybrid_live_result.json",
    "current_alert.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run live hybrid flood prediction for AP districts."
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Google Earth Engine project ID.",
    )
    parser.add_argument(
        "--district",
        action="append",
        default=[],
        help=(
            "District slug. Repeat to run multiple selected districts. "
            "Omit to run all districts."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected districts for testing.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip districts that already have a successful current_alert.json.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately instead of continuing after a district failure.",
    )
    return parser.parse_args()


def select_districts(args: argparse.Namespace) -> List[District]:
    if args.district:
        unknown = [
            slug for slug in args.district
            if slug not in DISTRICTS_BY_SLUG
        ]
        if unknown:
            valid = ", ".join(DISTRICTS_BY_SLUG)
            raise SystemExit(
                f"Unknown district slug(s): {unknown}\nValid slugs:\n{valid}"
            )
        selected = [
            DISTRICTS_BY_SLUG[slug] for slug in args.district
        ]
    else:
        selected = list(AP_DISTRICTS)

    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be at least 1")
        selected = selected[:args.limit]

    return selected


def existing_success(district: District) -> bool:
    path = (
        RESULTS_ROOT
        / district["slug"]
        / "current_alert.json"
    )
    if not path.exists():
        return False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False

    return payload.get("status") == "success"


def run_district(project: str, district: District) -> None:
    command = [
        sys.executable,
        "-m",
        "realtime.run_full_pipeline",
        "--project",
        project,
        "--lat",
        str(district["latitude"]),
        "--lon",
        str(district["longitude"]),
        "--name",
        district["slug"],
    ]

    print("\n" + "#" * 76)
    print(
        f"DISTRICT: {district['name']} "
        f"({district['slug']})"
    )
    print(
        "Coordinates:",
        district["latitude"],
        district["longitude"],
    )
    print("#" * 76)

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Full pipeline returned exit code {completed.returncode}"
        )


def archive_current_files(district: District) -> Path:
    district_dir = RESULTS_ROOT / district["slug"]
    district_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    for filename in FILES_TO_ARCHIVE:
        source = PROCESSED_DIR / filename
        destination = district_dir / filename

        if source.exists():
            shutil.copy2(source, destination)
            copied.append(filename)
        else:
            missing.append(filename)

    manifest = {
        "archived_utc": datetime.now(timezone.utc).isoformat(),
        "district_slug": district["slug"],
        "district_name": district["name"],
        "latitude": district["latitude"],
        "longitude": district["longitude"],
        "copied_files": copied,
        "missing_files": missing,
    }
    (district_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if "current_alert.json" in missing:
        raise FileNotFoundError(
            "The district run did not create current_alert.json"
        )

    return district_dir


def write_failure(district: District, exc: Exception) -> None:
    district_dir = RESULTS_ROOT / district["slug"]
    district_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "district_slug": district["slug"],
        "district_name": district["name"],
        "latitude": district["latitude"],
        "longitude": district["longitude"],
        "status": "failed",
        "error": str(exc),
    }
    (district_dir / "error.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    districts = select_districts(args)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("AP MULTI-DISTRICT LIVE HYBRID PIPELINE")
    print("=" * 76)
    print("Project        :", args.project)
    print("District count :", len(districts))
    print("Resume         :", args.resume)
    print("Results root   :", RESULTS_ROOT.relative_to(PROJECT_ROOT))

    completed_count = 0
    skipped_count = 0
    failed_count = 0

    for index, district in enumerate(districts, start=1):
        print(
            f"\n[{index}/{len(districts)}] "
            f"{district['name']}"
        )

        if args.resume and existing_success(district):
            print("  SKIPPED: successful archived result already exists")
            skipped_count += 1
            continue

        try:
            run_district(args.project, district)
            district_dir = archive_current_files(district)
            completed_count += 1
            print("  ARCHIVED:", district_dir.relative_to(PROJECT_ROOT))

            # Keep partial map outputs current after every successful district.
            build_outputs()

        except Exception as exc:
            failed_count += 1
            write_failure(district, exc)
            print(f"  FAILED: {exc}", file=sys.stderr)
            if args.stop_on_error:
                build_outputs()
                raise SystemExit(1) from exc

    summary = build_outputs()

    print("\n" + "=" * 76)
    print("MULTI-DISTRICT PIPELINE COMPLETE")
    print("=" * 76)
    print("Completed :", completed_count)
    print("Skipped   :", skipped_count)
    print("Failed    :", failed_count)
    print("Map-ready :", summary["successful_count"], "district results")
    print("JSON      : data\\processed\\ap_district_risk.json")
    print("CSV       : data\\processed\\ap_district_risk.csv")
    print("GeoJSON   : data\\processed\\ap_district_risk.geojson")

    if failed_count:
        print(
            "\nSome districts failed. Correct the errors and rerun with "
            "--resume; successful districts will be skipped."
        )


if __name__ == "__main__":
    main()
