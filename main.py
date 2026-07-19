"""Unified CLI for the current AP Flood Early Warning System.

Normal live operation loads the saved CNN/LSTM models; it does not retrain them.

Examples
--------
python main.py status
python main.py doctor
python main.py district --lat 16.51 --lon 80.65 --name ntr_vijayawada
python main.py all-districts --project ap-flood-monitor
python main.py refresh --project ap-flood-monitor
python main.py dashboard --port 5050
"""

from __future__ import annotations

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "outputs" / "reports"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
DISTRICT_RISK = PROCESSED / "ap_district_risk.json"
SMS_CONFIG = ROOT / "config" / "sms_config.json"


def make_logger(verbose: bool = False) -> logging.Logger:
    REPORTS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ap_flood_ews")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(REPORTS / "system.log", encoding="utf-8"),
    ):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


LOGGER = make_logger()


def execute(command: Sequence[str], label: str) -> None:
    LOGGER.info("=" * 68)
    LOGGER.info(label)
    LOGGER.info("Command: %s", " ".join(command))
    LOGGER.info("=" * 68)
    result = subprocess.run(list(command), cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def run_module(module_name: str, arguments: Iterable[str], label: str) -> None:
    execute([sys.executable, "-m", module_name, *list(arguments)], label)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def cmd_status(_args: argparse.Namespace) -> None:
    print("\nAP FLOOD EARLY WARNING SYSTEM — STATUS")
    print("=" * 68)
    print("Project root :", ROOT)
    print("Python       :", sys.executable)
    print("Python ver.  :", sys.version.split()[0])

    print("\nFinal artifacts")
    for name in (
        "real_cnn_model.h5",
        "real_lstm_model.h5",
        "cnn_threshold.json",
        "lstm_threshold.json",
        "lstm_scaler.joblib",
    ):
        path = MODELS / name
        state = f"{path.stat().st_size / 1_048_576:.2f} MB" if path.exists() else "MISSING"
        print(f"{name:<28} {state}")

    print("\nDistrict data")
    if DISTRICT_RISK.exists():
        data = read_json(DISTRICT_RISK)
        print("Generated    :", data.get("generated_utc", "unknown"))
        print("Total        :", data.get("district_count", "unknown"))
        print("Successful   :", data.get("successful_count", "unknown"))
        print("Unavailable  :", data.get("unavailable_count", "unknown"))
    else:
        print("Missing      :", DISTRICT_RISK.relative_to(ROOT))

    print("\nSMS")
    if SMS_CONFIG.exists():
        cfg = read_json(SMS_CONFIG)
        auto = cfg.get(
            "dashboard_auto_send_enabled",
            cfg.get("dashboard_auto_sms", False),
        )
        print("Provider     :", str(cfg.get("provider", "mock")).upper())
        print("Recipients   :", len(cfg.get("recipients", [])))
        print("Levels       :", ", ".join(cfg.get("alert_on_levels", [])))
        print("Automatic    :", bool(auto))
    else:
        print("Configuration missing; mock mode should be used.")

    print("\nAcademic prototype — not an official public warning service.")


def cmd_doctor(_args: argparse.Namespace) -> None:
    required_files = [
        ROOT / "dashboard" / "app.py",
        ROOT / "realtime" / "run_full_pipeline.py",
        ROOT / "realtime" / "run_all_districts.py",
        ROOT / "realtime" / "build_ap_risk_map_data.py",
        ROOT / "realtime" / "hybrid_decision.py",
        ROOT / "utils" / "sentinel_fetcher.py",
        ROOT / "utils" / "risk_map_generator.py",
        ROOT / "utils" / "alert_system.py",
        MODELS / "real_cnn_model.h5",
        MODELS / "real_lstm_model.h5",
        MODELS / "lstm_scaler.joblib",
    ]
    modules = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "tensorflow": "tensorflow",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
        "requests": "requests",
        "ee": "earthengine-api",
        "matplotlib": "matplotlib",
        "folium": "folium",
        "plotly": "plotly",
        "dash": "dash",
        "flask": "Flask",
        "colorama": "colorama",
    }

    failed = False
    print("\nRequired files")
    print("-" * 68)
    for path in required_files:
        ok = path.exists()
        failed = failed or not ok
        print(f"[{'OK' if ok else 'MISSING':<7}] {path.relative_to(ROOT)}")

    print("\nPython packages")
    print("-" * 68)
    for module_name, package_name in modules.items():
        try:
            importlib.import_module(module_name)
            try:
                installed_version = package_version(package_name)
            except PackageNotFoundError:
                installed_version = "installed"
            print(
                f"[OK     ] {package_name:<20} "
                f"{installed_version}"
            )
        except Exception as exc:
            failed = True
            print(f"[MISSING] {package_name:<20} {exc}")

    if failed:
        print("\nCHECK FAILED")
        print("Install packages with: python -m pip install -r requirements.txt")
        raise SystemExit(1)
    print("\nCHECK PASSED")


def cmd_district(args: argparse.Namespace) -> None:
    values = [
        "--project", args.project,
        "--lat", str(args.lat),
        "--lon", str(args.lon),
        "--name", args.name,
    ]
    if args.skip_sentinel_fetch:
        values.append("--skip-sentinel-fetch")
    if args.skip_weather_fetch:
        values.append("--skip-weather-fetch")
    run_module("realtime.run_full_pipeline", values, f"Live pipeline: {args.name}")


def all_district_arguments(args: argparse.Namespace) -> list[str]:
    values = ["--project", args.project]
    if args.resume:
        values.append("--resume")
    if args.limit is not None:
        values.extend(["--limit", str(args.limit)])
    for district in args.district:
        values.extend(["--district", district])
    if args.stop_on_error:
        values.append("--stop-on-error")
    return values


def cmd_all_districts(args: argparse.Namespace) -> None:
    run_module(
        "realtime.run_all_districts",
        all_district_arguments(args),
        "Run AP district live pipelines",
    )


def cmd_aggregate(_args: argparse.Namespace) -> None:
    run_module("realtime.build_ap_risk_map_data", [], "Build AP district risk data")


def cmd_map(_args: argparse.Namespace) -> None:
    run_module("utils.risk_map_generator", [], "Generate live AP risk maps")


def cmd_alert(args: argparse.Namespace) -> None:
    values = ["--mode", args.mode]
    if args.send_sms:
        values.append("--send-sms")
    run_module("utils.alert_system", values, f"Alert system: {args.mode}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    if args.prepare:
        cmd_aggregate(args)
        cmd_map(args)
        cmd_alert(argparse.Namespace(mode="live", send_sms=False))
    command = [
        sys.executable,
        str(ROOT / "dashboard" / "app.py"),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.debug:
        command.append("--debug")
    execute(command, "Start 26-district dashboard")


def cmd_refresh(args: argparse.Namespace) -> None:
    cmd_all_districts(args)
    cmd_aggregate(args)
    if not args.skip_map:
        cmd_map(args)
    if not args.skip_alert:
        cmd_alert(argparse.Namespace(mode="live", send_sms=False))
    print("\nRefresh complete. Start the dashboard with:")
    print("python main.py dashboard --port 5050")


def cmd_simple_module(args: argparse.Namespace) -> None:
    run_module(args.module, [], args.label)


def add_all_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default="ap-flood-monitor")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--district", action="append", default=[])
    parser.add_argument("--stop-on-error", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freshness-aware CNN + LSTM AP Flood EWS",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="Show current models, results, and SMS mode")
    p.set_defaults(handler=cmd_status)

    p = sub.add_parser("doctor", help="Validate files and Python packages")
    p.set_defaults(handler=cmd_doctor)

    p = sub.add_parser("district", help="Run one live district/location")
    p.add_argument("--project", default="ap-flood-monitor")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--skip-sentinel-fetch", action="store_true")
    p.add_argument("--skip-weather-fetch", action="store_true")
    p.set_defaults(handler=cmd_district)

    p = sub.add_parser("all-districts", help="Run all or selected AP districts")
    add_all_options(p)
    p.set_defaults(handler=cmd_all_districts)

    p = sub.add_parser("aggregate", help="Build district JSON/CSV/GeoJSON")
    p.set_defaults(handler=cmd_aggregate)

    p = sub.add_parser("map", help="Generate interactive and static risk maps")
    p.set_defaults(handler=cmd_map)

    p = sub.add_parser("alert", help="Run live alert, SMS test, or phone validation")
    p.add_argument("--mode", choices=("live", "sms-test", "validate-phones"), default="live")
    p.add_argument("--send-sms", action="store_true")
    p.set_defaults(handler=cmd_alert)

    p = sub.add_parser("dashboard", aliases=["serve"], help="Start the Dash dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.set_defaults(handler=cmd_dashboard)

    p = sub.add_parser("refresh", aliases=["pipeline"], help="Districts -> aggregate -> map -> alert")
    add_all_options(p)
    p.add_argument("--skip-map", action="store_true")
    p.add_argument("--skip-alert", action="store_true")
    p.set_defaults(handler=cmd_refresh)

    development = {
        "train-cnn": ("models.train_real_cnn", "Train CNN"),
        "evaluate-cnn": ("models.evaluate_real_cnn", "Evaluate CNN"),
        "build-lstm-data": ("utils.build_real_lstm_dataset", "Build LSTM dataset"),
        "train-lstm": ("models.train_real_lstm", "Train LSTM"),
        "evaluate-lstm": ("models.evaluate_real_lstm", "Evaluate LSTM"),
    }
    for command, (module, label) in development.items():
        p = sub.add_parser(command, help=f"Optional model-development command: {label}")
        p.set_defaults(handler=cmd_simple_module, module=module, label=label)

    return parser


def main() -> None:
    global LOGGER
    parser = build_parser()
    args = parser.parse_args()
    LOGGER = make_logger(args.verbose)
    try:
        args.handler(args)
    except KeyboardInterrupt:
        LOGGER.warning("Stopped by user")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception:
        LOGGER.exception("Command failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
