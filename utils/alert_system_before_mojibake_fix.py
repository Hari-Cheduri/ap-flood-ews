"""
Live district-level AP flood alert and SMS system.

This version consumes data/processed/ap_district_risk.json generated from the
real district CNN + LSTM fusion pipeline. It does not create simulated 50×50
risk grids and does not apply a second monsoon multiplier.

Default behaviour is safe:
- GREEN/YELLOW: dashboard and log only
- ORANGE/RED: eligible for SMS
- SMS is sent only when --send-sms is explicitly supplied
- The default SMS provider remains mock

Usage
-----
Evaluate and display the latest district alerts:
    python -m utils.alert_system --mode live

Evaluate and dispatch configured ORANGE/RED SMS:
    python -m utils.alert_system --mode live --send-sms

Test SMS configuration:
    python -m utils.alert_system --mode sms-test
"""

from __future__ import annotations

import sys

# Prevent scheduled/background execution from crashing when Windows uses
# a legacy console encoding such as CP1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

import sys

# Prevent scheduled/background execution from crashing when Windows uses
# a legacy console encoding such as CP1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

import argparse
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from colorama import Fore, Style
from colorama import init as colorama_init

colorama_init(autoreset=True)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RISK_PATH = (
    PROJECT_ROOT / "data" / "processed" / "ap_district_risk.json"
)
DEFAULT_LOG_PATH = (
    PROJECT_ROOT / "outputs" / "predictions" / "alert_log.csv"
)

ALERT_LEVELS: Dict[str, Dict[str, Any]] = {
    "GREEN": {
        "priority": 0,
        "color": Fore.GREEN,
        "action": "No public alert. Continue routine monitoring.",
    },
    "YELLOW": {
        "priority": 1,
        "color": Fore.YELLOW,
        "action": (
            "Watch conditions and verify district field reports. "
            "No evacuation instruction."
        ),
    },
    "ORANGE": {
        "priority": 2,
        "color": Fore.YELLOW + Style.BRIGHT,
        "action": (
            "Verify river/drainage conditions and prepare district response. "
            "Notify authorities after verification."
        ),
    },
    "RED": {
        "priority": 3,
        "color": Fore.RED + Style.BRIGHT,
        "action": (
            "Urgent official verification required. Activate emergency "
            "protocols only through authorised authorities."
        ),
    },
    "UNKNOWN": {
        "priority": -1,
        "color": Fore.WHITE,
        "action": "Prediction unavailable.",
    },
}


class FloodAlertSystem:
    """Evaluate the latest archived district hybrid results."""

    LOG_FIELDS = [
        "timestamp",
        "state_level",
        "available_districts",
        "green_count",
        "yellow_count",
        "orange_count",
        "red_count",
        "max_fusion_score",
        "top_district",
        "districts_at_risk",
        "message",
    ]

    def __init__(
        self,
        risk_path: str | Path = DEFAULT_RISK_PATH,
        log_path: str | Path = DEFAULT_LOG_PATH,
    ) -> None:
        self.risk_path = Path(risk_path)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_log()

    def _init_log(self) -> None:
        if self.log_path.exists():
            try:
                existing = pd.read_csv(
                    self.log_path,
                    nrows=0,
                ).columns.tolist()
                if existing != self.LOG_FIELDS:
                    backup = self.log_path.with_suffix(
                        f".legacy_{datetime.now():%Y%m%d_%H%M%S}.csv"
                    )
                    self.log_path.replace(backup)
            except Exception:
                backup = self.log_path.with_suffix(
                    f".invalid_{datetime.now():%Y%m%d_%H%M%S}.csv"
                )
                self.log_path.replace(backup)

        if not self.log_path.exists():
            with self.log_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                csv.DictWriter(
                    handle,
                    fieldnames=self.LOG_FIELDS,
                ).writeheader()

    def _load_payload(self) -> Dict[str, Any]:
        if not self.risk_path.exists():
            raise FileNotFoundError(
                f"Missing {self.risk_path}. Run "
                "'python -m realtime.build_ap_risk_map_data' first."
            )
        return json.loads(
            self.risk_path.read_text(encoding="utf-8")
        )

    @staticmethod
    def _district_level(record: Dict[str, Any]) -> str:
        level = str(record.get("alert_level") or "UNKNOWN").upper()
        return level if level in ALERT_LEVELS else "UNKNOWN"

    def evaluate_live_results(self) -> Dict[str, Any]:
        payload = self._load_payload()
        districts = payload.get("districts") or []

        available = [
            dict(record)
            for record in districts
            if record.get("status") == "success"
        ]
        for record in available:
            record["alert_level"] = self._district_level(record)

        counts = {
            level: sum(
                record["alert_level"] == level
                for record in available
            )
            for level in ("GREEN", "YELLOW", "ORANGE", "RED")
        }

        ranked = sorted(
            available,
            key=lambda record: (
                ALERT_LEVELS[record["alert_level"]]["priority"],
                float(record.get("fusion_score") or 0.0),
            ),
            reverse=True,
        )

        top = ranked[0] if ranked else None
        state_level = (
            top["alert_level"] if top is not None else "UNKNOWN"
        )

        at_risk = [
            record
            for record in ranked
            if record["alert_level"] in {"YELLOW", "ORANGE", "RED"}
        ]

        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state_level": state_level,
            "level": state_level,
            "available_districts": len(available),
            "total_districts": len(districts),
            "district_counts": counts,
            "green_count": counts["GREEN"],
            "yellow_count": counts["YELLOW"],
            "orange_count": counts["ORANGE"],
            "red_count": counts["RED"],
            "max_fusion_score": (
                float(top.get("fusion_score") or 0.0)
                if top is not None
                else None
            ),
            "top_district": (
                top.get("district_name") if top is not None else None
            ),
            "districts_at_risk": [
                {
                    "district_slug": record.get("district_slug"),
                    "district_name": record.get("district_name"),
                    "alert_level": record.get("alert_level"),
                    "fusion_score": record.get("fusion_score"),
                    "cnn_probability": record.get("cnn_probability"),
                    "cnn_confidence": record.get("cnn_confidence"),
                    "lstm_probability": record.get("lstm_probability"),
                    "lstm_confidence": record.get("lstm_confidence"),
                    "rain_last_24h_mm": record.get("rain_last_24h_mm"),
                    "rain_last_72h_mm": record.get("rain_last_72h_mm"),
                    "message": record.get("message"),
                }
                for record in at_risk
            ],
            "message": ALERT_LEVELS[state_level]["action"],
            "source_generated_utc": payload.get("generated_utc"),
            "limitation": (
                "Academic district-level prototype. Alerts must be verified "
                "using official river, rainfall, field, and disaster-management "
                "information before public action."
            ),
        }
        return alert

    def log_alert(self, alert: Dict[str, Any]) -> None:
        district_names = ", ".join(
            (
                f"{item.get('district_name')} "
                f"({item.get('alert_level')})"
            )
            for item in alert.get("districts_at_risk", [])
        )

        row = {
            "timestamp": alert.get("timestamp"),
            "state_level": alert.get("state_level"),
            "available_districts": alert.get("available_districts"),
            "green_count": alert.get("green_count"),
            "yellow_count": alert.get("yellow_count"),
            "orange_count": alert.get("orange_count"),
            "red_count": alert.get("red_count"),
            "max_fusion_score": alert.get("max_fusion_score"),
            "top_district": alert.get("top_district"),
            "districts_at_risk": district_names,
            "message": alert.get("message"),
        }

        with self.log_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.LOG_FIELDS,
            )
            writer.writerow(row)

    def display_alert(self, alert: Dict[str, Any]) -> None:
        level = str(alert.get("state_level", "UNKNOWN"))
        config = ALERT_LEVELS.get(level, ALERT_LEVELS["UNKNOWN"])
        color = config["color"]
        bar = "█" * 70

        print(f"\n{color}{bar}")
        print(f"  AP LIVE HYBRID FLOOD MONITOR — {level}")
        print(f"  {alert.get('timestamp')}")
        print(f"{color}{bar}")
        print(
            f"  District results : "
            f"{alert.get('available_districts')}/"
            f"{alert.get('total_districts')}"
        )
        print(
            "  Counts           : "
            f"GREEN={alert.get('green_count')}  "
            f"YELLOW={alert.get('yellow_count')}  "
            f"ORANGE={alert.get('orange_count')}  "
            f"RED={alert.get('red_count')}"
        )

        max_score = alert.get("max_fusion_score")
        if isinstance(max_score, (int, float)):
            print(f"  Highest fusion   : {max_score * 100:.1f}%")
        print(f"  Top district     : {alert.get('top_district') or 'Unavailable'}")

        at_risk = alert.get("districts_at_risk") or []
        if at_risk:
            print("  District watch list:")
            for record in at_risk[:12]:
                score = record.get("fusion_score")
                score_text = (
                    f"{float(score) * 100:.1f}%"
                    if isinstance(score, (int, float))
                    else "Unavailable"
                )
                print(
                    f"    {record.get('alert_level', '?'):<7} "
                    f"{record.get('district_name', '?'):<32} "
                    f"{score_text:>10}"
                )
        else:
            print("  District watch list: none")

        print(f"  Action           : {alert.get('message')}")
        print(f"{color}{bar}{Style.RESET_ALL}\n")

    def process_live_results(
        self,
        send_sms: bool = False,
        sms_config: str = "config/sms_config.json",
    ) -> Dict[str, Any]:
        alert = self.evaluate_live_results()
        self.display_alert(alert)
        self.log_alert(alert)

        if send_sms:
            SMSAlertSystem(
                config_path=sms_config
            ).send_alert_sms(alert)
        else:
            print(
                "[SMS] Not requested. Add --send-sms to dispatch eligible "
                "ORANGE/RED alerts."
            )

        return alert

    def get_alert_history(self, n: int = 50) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.log_path).tail(n)


class SMSAlertSystem:
    """Twilio, Fast2SMS, or mock SMS dispatch for live district alerts."""

    DEFAULT_CONFIG = {
        "provider": "mock",
        "twilio": {
            "account_sid": "YOUR_TWILIO_SID",
            "auth_token": "YOUR_TWILIO_TOKEN",
            "from_number": "+1XXXXXXXXXX",
        },
        "fast2sms": {
            "api_key": "YOUR_FAST2SMS_KEY",
            "sender_id": "FLOODS",
        },
        "recipients": [
            {
                "name": "AP District Authority",
                "phone": "+919XXXXXXXXX",
            }
        ],
        "alert_on_levels": ["ORANGE", "RED"],
        "cooldown_minutes": 30,
        "max_sms_per_hour": 10,
    }

    def __init__(
        self,
        config_path: str = "config/sms_config.json",
    ) -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        self.provider = str(
            self.config.get("provider", "mock")
        ).lower()
        self.log_path = (
            PROJECT_ROOT
            / "outputs"
            / "predictions"
            / "sms_log.csv"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = None
        self.from_number = None
        self.fast2sms_key = None
        self._init_provider()

    def _load_config(self, path: Path) -> Dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                json.dumps(self.DEFAULT_CONFIG, indent=2),
                encoding="utf-8",
            )
            print(f"[SMS] Created safe mock config at {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _init_provider(self) -> None:
        if self.provider == "twilio":
            from twilio.rest import Client  # type: ignore

            config = self.config.get("twilio") or {}
            self.client = Client(
                config.get("account_sid"),
                config.get("auth_token"),
            )
            self.from_number = config.get("from_number")
            print("[SMS] Twilio provider initialised")
        elif self.provider == "fast2sms":
            config = self.config.get("fast2sms") or {}
            self.fast2sms_key = config.get("api_key")
            print("[SMS] Fast2SMS provider initialised")
        else:
            self.provider = "mock"
            print("[SMS] Mock mode — no real message will be transmitted")

    @staticmethod
    def _placeholder_phone(phone: str) -> bool:
        upper = phone.upper()
        digits = "".join(character for character in phone if character.isdigit())
        return "X" in upper or len(digits) < 10

    def _format_message(self, alert: Dict[str, Any]) -> str:
        districts = alert.get("districts_at_risk") or []
        urgent = [
            record
            for record in districts
            if record.get("alert_level") in {"ORANGE", "RED"}
        ]
        district_text = ", ".join(
            (
                f"{record.get('district_name')} "
                f"{record.get('alert_level')} "
                f"{float(record.get('fusion_score') or 0.0) * 100:.0f}%"
            )
            for record in urgent[:5]
        ) or "No ORANGE/RED district listed"

        return (
            f"[AP FLOOD EWS - {alert.get('state_level')}]\n"
            f"Districts: {district_text}\n"
            f"Highest: {float(alert.get('max_fusion_score') or 0.0) * 100:.1f}%\n"
            f"Time: {str(alert.get('timestamp'))[:19]}\n"
            f"Action: {alert.get('message')}\n"
            f"Verify with official field/river data before public action."
        )

    def _read_sms_log(self) -> List[Dict[str, str]]:
        if not self.log_path.exists():
            return []
        with self.log_path.open(
            newline="",
            encoding="utf-8",
        ) as handle:
            return list(csv.DictReader(handle))

    def _check_cooldown(self, phone: str) -> bool:
        cutoff = datetime.now() - timedelta(
            minutes=int(self.config.get("cooldown_minutes", 30))
        )
        for row in self._read_sms_log():
            if row.get("phone") != phone:
                continue
            try:
                timestamp = datetime.fromisoformat(
                    str(row.get("timestamp"))
                )
            except ValueError:
                continue
            if timestamp > cutoff:
                return False
        return True

    def _hourly_count(self) -> int:
        cutoff = datetime.now() - timedelta(hours=1)
        count = 0
        for row in self._read_sms_log():
            try:
                timestamp = datetime.fromisoformat(
                    str(row.get("timestamp"))
                )
            except ValueError:
                continue
            status = str(row.get("status", ""))
            if timestamp > cutoff and (
                status.startswith("sent:")
                or status == "mock_sent"
            ):
                count += 1
        return count

    def _log_sms(
        self,
        phone: str,
        name: str,
        message: str,
        status: str,
    ) -> None:
        fields = [
            "timestamp",
            "name",
            "phone",
            "provider",
            "status",
            "message",
        ]
        write_header = not self.log_path.exists()
        with self.log_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
            )
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "name": name,
                "phone": phone,
                "provider": self.provider,
                "status": status,
                "message": message[:120],
            })

    def send_twilio(self, phone: str, message: str) -> str:
        if self.client is None:
            raise RuntimeError("Twilio client is not initialised")
        result = self.client.messages.create(
            body=message,
            from_=self.from_number,
            to=phone,
        )
        return str(result.sid)

    def send_fast2sms(self, phone: str, message: str) -> str:
        import requests

        number = phone.replace("+91", "").replace("+", "").strip()
        response = requests.post(
            "https://www.fast2sms.com/dev/bulkV2",
            json={
                "route": "q",
                "message": message,
                "language": "english",
                "flash": 0,
                "numbers": number,
            },
            headers={
                "authorization": str(self.fast2sms_key),
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        response.raise_for_status()
        return str(response.json().get("request_id", "OK"))

    def send_alert_sms(
        self,
        alert: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        level = str(alert.get("state_level", "GREEN"))
        eligible = self.config.get(
            "alert_on_levels",
            ["ORANGE", "RED"],
        )
        if level not in eligible:
            print(f"[SMS] {level} is not eligible — skipped")
            return []

        max_per_hour = int(
            self.config.get("max_sms_per_hour", 10)
        )
        if self._hourly_count() >= max_per_hour:
            print(f"[SMS] Hourly cap {max_per_hour} reached — skipped")
            return []

        message = self._format_message(alert)
        results = []

        for recipient in self.config.get("recipients", []):
            name = str(recipient.get("name", "Recipient"))
            phone = str(recipient.get("phone", ""))

            if self._placeholder_phone(phone):
                print(f"[SMS] Placeholder/invalid phone for {name} — skipped")
                results.append({
                    "name": name,
                    "phone": phone,
                    "status": "invalid_phone",
                })
                continue

            if not self._check_cooldown(phone):
                print(f"[SMS] Cooldown active for {name} — skipped")
                results.append({
                    "name": name,
                    "phone": phone,
                    "status": "cooldown",
                })
                continue

            try:
                if self.provider == "twilio":
                    reference = self.send_twilio(phone, message)
                    status = f"sent:{reference}"
                elif self.provider == "fast2sms":
                    reference = self.send_fast2sms(phone, message)
                    status = f"sent:{reference}"
                else:
                    print("\n" + "=" * 58)
                    print(f"[MOCK SMS] {name} ({phone})")
                    print("-" * 58)
                    print(message)
                    print("=" * 58)
                    status = "mock_sent"

                self._log_sms(phone, name, message, status)
                results.append({
                    "name": name,
                    "phone": phone,
                    "status": status,
                })
            except Exception as exc:
                status = f"error:{exc}"
                self._log_sms(phone, name, message, status)
                results.append({
                    "name": name,
                    "phone": phone,
                    "status": status,
                })
                print(f"[SMS] {name} failed: {exc}")

        return results

    def test_connection(self) -> List[Dict[str, Any]]:
        test_alert = {
            "state_level": "ORANGE",
            "level": "ORANGE",
            "max_fusion_score": 0.72,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "TEST ONLY — connectivity check.",
            "districts_at_risk": [
                {
                    "district_name": "TEST DISTRICT",
                    "alert_level": "ORANGE",
                    "fusion_score": 0.72,
                }
            ],
        }

        original = self.config.get("cooldown_minutes", 30)
        self.config["cooldown_minutes"] = 0
        try:
            return self.send_alert_sms(test_alert)
        finally:
            self.config["cooldown_minutes"] = original

    def validate_phone_numbers(self) -> Dict[str, bool]:
        try:
            import phonenumbers  # type: ignore
        except ImportError:
            print("Install with: pip install phonenumbers")
            return {}

        results: Dict[str, bool] = {}
        for recipient in self.config.get("recipients", []):
            name = str(recipient.get("name"))
            phone = str(recipient.get("phone"))
            key = f"{name} ({phone})"

            if self._placeholder_phone(phone):
                results[key] = False
                print(f"[SMS] invalid placeholder: {key}")
                continue

            try:
                parsed = phonenumbers.parse(phone, None)
                valid = phonenumbers.is_valid_number(parsed)
            except phonenumbers.NumberParseException:
                valid = False

            results[key] = valid
            print(f"[SMS] {key}: valid={valid}")

        return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["live", "sms-test", "validate-phones"],
        default="live",
    )
    parser.add_argument(
        "--risk-data",
        default=str(DEFAULT_RISK_PATH),
    )
    parser.add_argument(
        "--log",
        default=str(DEFAULT_LOG_PATH),
    )
    parser.add_argument(
        "--sms-config",
        default="config/sms_config.json",
    )
    parser.add_argument(
        "--send-sms",
        action="store_true",
        help="Explicitly dispatch eligible ORANGE/RED alerts.",
    )
    args = parser.parse_args()

    if args.mode == "live":
        system = FloodAlertSystem(
            risk_path=args.risk_data,
            log_path=args.log,
        )
        system.process_live_results(
            send_sms=args.send_sms,
            sms_config=args.sms_config,
        )
    elif args.mode == "sms-test":
        SMSAlertSystem(
            config_path=args.sms_config
        ).test_connection()
    else:
        SMSAlertSystem(
            config_path=args.sms_config
        ).validate_phone_numbers()


if __name__ == "__main__":
    main()

