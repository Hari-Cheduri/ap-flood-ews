"""
utils/alert_system.py
─────────────────────────────────────────────────────────────────────────────
Flood alert evaluation, tiered console display, CSV logging, real-time stream
simulation, and SMS dispatching for the Andhra Pradesh river-basin early-
warning system.

Covers four major river basins:
  Vamsadhara (18.35°N) · Godavari (17.25°N) · Krishna (16.15°N) · Penna (14.45°N)

Classes
-------
FloodAlertSystem  – evaluates risk grids, logs alerts, streams simulation
SMSAlertSystem    – sends/mocks SMS via Twilio (primary) or Fast2SMS (India
                    fallback), with per-recipient cooldown and CSV audit log
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from colorama import Fore, Style
from colorama import init as colorama_init

colorama_init(autoreset=True)


# ── Constants ─────────────────────────────────────────────────────────────────

ALERT_LEVELS: dict[str, dict[str, Any]] = {
    "GREEN": {
        "min": 0.00, "max": 0.35,
        "color": Fore.GREEN,
        "action": "No action required. Monitor conditions.",
    },
    "YELLOW": {
        "min": 0.35, "max": 0.55,
        "color": Fore.YELLOW,
        "action": "Precautionary alert. Prepare evacuation plans.",
    },
    "ORANGE": {
        "min": 0.55, "max": 0.75,
        "color": Fore.YELLOW + Style.BRIGHT,
        "action": "High risk. Begin voluntary evacuations.",
    },
    "RED": {
        "min": 0.75, "max": 1.01,
        "color": Fore.RED + Style.BRIGHT,
        "action": "CRITICAL. Mandatory evacuation. Deploy NDRF.",
    },
}

# ── Andhra Pradesh river system ───────────────────────────────────────────────

# Centre latitudes of the four major AP rivers
GODAVARI_LAT   = 17.25
KRISHNA_LAT    = 16.15
PENNA_LAT      = 14.45
VAMSADHARA_LAT = 18.35

# Bounding box used to map risk-grid rows → geographic latitude
AP_LAT_MIN = 12.5    # matches risk_map_generator.py LAT_MIN
AP_LAT_MAX = 19.5    # matches risk_map_generator.py LAT_MAX

# Flood-prone zones organised by river basin.
# Tuple layout: (display_name, river_label, centre_lat°N, centre_lon°E)
AP_FLOOD_ZONES: list[tuple[str, str, float, float]] = [
    # ── Vamsadhara basin (18.35°N) ────────────────────────────────────────────
    ("Srikakulam District",          "Vamsadhara", 18.30, 83.90),
    ("Narasannapeta Corridor",       "Vamsadhara", 18.42, 83.63),
    # ── Godavari basin (17.25°N) ──────────────────────────────────────────────
    ("Rajamahendravaram Riverfront",  "Godavari",  17.01, 81.77),
    ("Eluru West Godavari",           "Godavari",  16.71, 81.10),
    ("Amalapuram Delta",              "Godavari",  16.58, 82.01),
    ("Bhimavaram Lowlands",           "Godavari",  16.54, 81.52),
    # ── Krishna basin (16.15°N) ───────────────────────────────────────────────
    ("Vijayawada Floodplain",         "Krishna",   16.51, 80.62),
    ("Guntur District Lowlands",      "Krishna",   16.30, 80.46),
    ("Machilipatnam Delta",           "Krishna",   16.17, 81.13),
    ("Tenali Basin",                  "Krishna",   16.24, 80.64),
    # ── Penna basin (14.45°N) ─────────────────────────────────────────────────
    ("Nellore Urban Corridor",        "Penna",     14.44, 79.99),
    ("Kavali Coastal Stretch",        "Penna",     14.92, 80.02),
    ("Ongole Floodplain",             "Penna",     15.50, 80.05),
]

# Month → monsoon risk multiplier (1 = January … 12 = December)
# AP is doubly affected:
#   SW monsoon  → June–September  (primary; drives Godavari / Krishna flooding)
#   NE monsoon  → October–November (significant coastal AP; cyclone season)
MONSOON_MULTIPLIER: dict[int, float] = {
    1:  0.3,   # dry post-winter
    2:  0.3,   # dry
    3:  0.4,   # pre-summer
    4:  0.5,   # hot; occasional thunderstorms
    5:  0.6,   # pre-monsoon showers
    6:  1.1,   # SW monsoon onset
    7:  1.4,   # SW monsoon — major river-basin inflows
    8:  1.5,   # SW monsoon peak — Godavari / Krishna floods
    9:  1.3,   # SW monsoon withdrawal
    10: 1.2,   # NE monsoon begins; coastal surge risk rises
    11: 1.4,   # NE monsoon peak — cyclone season (highest coastal risk)
    12: 0.6,   # NE monsoon withdrawal
}


def _lat_to_row(lat: float, grid_h: int = 50) -> int:
    """
    Convert a geographic latitude (°N) to a risk-grid row index.

    Row 0 is the northernmost row (AP_LAT_MAX); row grid_h-1 is the
    southernmost row (AP_LAT_MIN).  Clipped to valid range.
    """
    frac = (AP_LAT_MAX - lat) / (AP_LAT_MAX - AP_LAT_MIN)
    return int(np.clip(frac * grid_h, 0, grid_h - 1))


# ══════════════════════════════════════════════════════════════════════════════
# FloodAlertSystem
# ══════════════════════════════════════════════════════════════════════════════

class FloodAlertSystem:
    """
    Evaluates 2-D flood probability grids, issues tiered alerts with
    colourised console output, logs to CSV, and optionally runs a
    simulated real-time stream that integrates SMS dispatching.

    Parameters
    ----------
    log_path  : path of the CSV alert log
    demo_mode : when True, the simulated stream visits all four alert
                tiers in order (GREEN → YELLOW → ORANGE → RED) so that
                every level is demonstrated in a single run.
    """

    LOG_FIELDS = [
        "timestamp", "level", "max_risk", "mean_risk",
        "affected_area_km2", "zones_at_risk", "message",
    ]

    def __init__(
        self,
        log_path: str = "outputs/predictions/alert_log.csv",
        demo_mode: bool = False,
    ) -> None:
        self.log_path  = log_path
        self.demo_mode = demo_mode
        _d = os.path.dirname(log_path)
        if _d:
            os.makedirs(_d, exist_ok=True)
        self._init_log()

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _init_log(self) -> None:
        """Create the alert-log CSV, or recreate it if its columns are stale."""
        if os.path.exists(self.log_path):
            try:
                existing_cols = pd.read_csv(self.log_path, nrows=0).columns.tolist()
                if existing_cols != self.LOG_FIELDS:
                    os.remove(self.log_path)   # stale schema — wipe and recreate
            except Exception:
                os.remove(self.log_path)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=self.LOG_FIELDS).writeheader()

    def log_alert(self, alert: dict) -> None:
        """Append one alert record to the CSV log."""
        with open(self.log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.LOG_FIELDS)
            w.writerow({k: alert.get(k, "") for k in self.LOG_FIELDS})

    # ── Risk evaluation ───────────────────────────────────────────────────────

    def evaluate_risk_map(
        self,
        risk_grid: np.ndarray,
        grid_res_km: float = 0.5,
    ) -> dict:
        """
        Derive a structured alert dict from a 2-D probability grid.

        Parameters
        ----------
        risk_grid   : float32 array of shape (H, W) with values in [0, 1]
        grid_res_km : spatial resolution of each cell in kilometres
        """
        max_risk  = float(np.max(risk_grid))
        mean_risk = float(np.mean(risk_grid))

        cell_area_km2  = grid_res_km ** 2
        n_high_cells   = int(np.sum(risk_grid >= ALERT_LEVELS["YELLOW"]["min"]))
        affected_area  = n_high_cells * cell_area_km2

        month      = datetime.now().month
        multiplier = MONSOON_MULTIPLIER[month]
        # Apply seasonal multiplier to mean risk for classification
        adjusted_mean = min(1.0, float(mean_risk * multiplier))

        # Level is determined by mean_risk across the full grid.
        # Using max_risk caused every step to fire RED: the single
        # highest cell in 2,500 almost always exceeds 0.75 even when
        # the regional mean is only 0.20 (GREEN territory).
        level = "GREEN"
        for lvl, cfg in ALERT_LEVELS.items():
            if cfg["min"] <= adjusted_mean < cfg["max"]:
                level = lvl
                break

        zones      = self._identify_zones(risk_grid, current_level=level)

        return {
            "timestamp":          datetime.now().isoformat(),
            "level":              level,
            "max_risk":           max_risk,
            "mean_risk":          mean_risk,
            "affected_area_km2":  affected_area,
            "zones_at_risk":      ", ".join(zones),
            "message":            ALERT_LEVELS[level]["action"],
            "monsoon_factor":     multiplier,
            "adjusted_mean_risk": round(adjusted_mean, 4),
        }

    def _identify_zones(
        self,
        risk_grid: np.ndarray,
        current_level: str = "ORANGE",
    ) -> list[str]:
        """
        Map each zone in AP_FLOOD_ZONES to its corresponding row-band in the
        risk grid using geographic latitude, then return the zones whose
        band-mean risk meets or exceeds the threshold for the current alert
        level (floored at YELLOW so zones are never suppressed below the
        lowest meaningful tier).

        Zone labels include the river name in brackets so callers can group
        results by basin, e.g. ``"Vijayawada Floodplain [Krishna]"``.
        """
        h, _  = risk_grid.shape
        band  = max(1, h // 25)   # ±2 rows for a 50-row grid
        zones: list[str] = []

        for name, river, clat, _ in AP_FLOOD_ZONES:
            row      = _lat_to_row(clat, h)
            r0       = max(0, row - band)
            r1       = min(h, row + band + 1)
            zone_max = float(np.mean(risk_grid[r0:r1, :]))
            # Use the minimum threshold of the current level (not always ORANGE)
            level_min = ALERT_LEVELS.get(current_level, ALERT_LEVELS["ORANGE"])["min"]
            if zone_max >= max(level_min, ALERT_LEVELS["YELLOW"]["min"]):
                zones.append(f"{name} [{river}]")

        return zones or ["No zones above threshold"]

    # ── Console display ───────────────────────────────────────────────────────

    def display_alert(self, alert: dict) -> None:
        """
        Print a colourised alert banner with a river-basin breakdown panel.

        Zones stored as ``"Name [River]"`` strings are grouped by river so
        the operator immediately sees which basins are at risk.
        """
        level = alert["level"]
        color = ALERT_LEVELS[level]["color"]
        bar   = "█" * 64
        ts    = str(alert.get("timestamp", ""))[:19]

        # ── Group at-risk zones by river basin ────────────────────────────────
        river_groups: dict[str, list[str]] = {}
        raw_zones = str(alert.get("zones_at_risk", ""))
        for entry in raw_zones.split(","):
            entry = entry.strip()
            if not entry or entry == "No zones above threshold":
                continue
            if "[" in entry:
                zone_name = entry.split("[")[0].strip()
                river     = entry.split("[")[1].rstrip("]").strip()
            else:
                zone_name, river = entry, "Other"
            river_groups.setdefault(river, []).append(zone_name)

        print(f"\n{color}{bar}")
        print(f"  ⚠  ANDHRA PRADESH FLOOD EARLY WARNING — {level} ALERT")
        print(f"  {ts}")
        print(f"{color}{bar}")
        print(f"  Max Risk  : {alert['max_risk']*100:.1f}%")
        print(f"  Mean Risk : {alert['mean_risk']*100:.1f}%")
        print(f"  Affected  : {alert['affected_area_km2']:.1f} km²")

        if river_groups:
            print(f"  River Basin Exposure:")
            river_order = ["Vamsadhara", "Godavari", "Krishna", "Penna"]
            for river in river_order:
                if river in river_groups:
                    names = ", ".join(river_groups[river])
                    print(f"    {river:<13s} ▶  {names}")
            for river, names_list in river_groups.items():
                if river not in river_order:
                    print(f"    {river:<13s} ▶  {', '.join(names_list)}")
        else:
            print(f"  Zones     : No zones above threshold")

        print(f"  Action    : {alert['message']}")
        print(f"{color}{bar}{Style.RESET_ALL}\n")

    # ── Real-time stream simulation ───────────────────────────────────────────

    def simulate_realtime_stream(
        self,
        n_steps:       int   = 12,
        interval_sec:  float = 2.0,
        seed:          str   = "AP_FLOOD",
        sms_config:    str   = "config/sms_config.json",
    ) -> None:
        """
        Simulate a time-series of flood risk updates across the AP river basins.

        Deterministic via MD5-seeding from `seed`.  In demo_mode the
        risk trajectory is forced to visit all four alert tiers so that
        GREEN → YELLOW → ORANGE → RED are all exercised in one run.

        An ``SMSAlertSystem`` is initialised once and called after every
        ``log_alert()``.  In mock mode (default config) it prints the
        message without hitting any real gateway.

        Parameters
        ----------
        n_steps      : number of time-steps to simulate
        interval_sec : sleep between steps (set 0 for instant runs)
        seed         : string used to derive the numpy RNG seed via MD5
        sms_config   : path forwarded to SMSAlertSystem
        """
        seed_int = int(hashlib.md5(seed.encode()).hexdigest(), 16) % (2 ** 32)
        rng      = np.random.default_rng(seed_int)

        print(f"\n{Fore.CYAN}{'─'*64}")
        print(f"  AP River-Basin Stream  |  steps={n_steps}  seed={seed}")
        print(f"{'─'*64}{Style.RESET_ALL}")

        # ── Build risk trajectory ─────────────────────────────────────────────
        if self.demo_mode:
            # Interpolate through all four alert anchors
            anchors    = [0.20, 0.45, 0.65, 0.85]
            x          = np.linspace(0, len(anchors) - 1, n_steps)
            xi         = np.arange(len(anchors))
            trajectory = np.interp(x, xi, anchors)
            trajectory += rng.uniform(-0.04, 0.04, n_steps)
        else:
            # Sigmoid-squeezed random walk
            walk       = np.cumsum(rng.normal(0, 0.08, n_steps))
            trajectory = 0.5 + 0.5 * np.tanh(walk)

        trajectory = np.clip(trajectory, 0.05, 0.99)

        month      = datetime.now().month
        multiplier = MONSOON_MULTIPLIER[month]

        # Initialise SMS once for the entire stream
        sms = SMSAlertSystem(config_path=sms_config)

        for step, base_risk in enumerate(trajectory, start=1):
            # Build 50 × 50 probability grid around the base risk
            grid = rng.beta(
                a=max(0.5, base_risk * 5),
                b=max(0.5, (1 - base_risk) * 5),
                size=(50, 50),
            ).astype(np.float32)
            grid = np.clip(grid * multiplier, 0.0, 1.0)

            alert = self.evaluate_risk_map(grid)
            self.display_alert(alert)
            self.log_alert(alert)

            # ── Send SMS ──────────────────────────────────────────────────────
            sms.send_alert_sms(alert)

            print(f"  Step {step}/{n_steps}  "
                  f"(monsoon factor: {multiplier:.2f}x)")

            if step < n_steps:
                time.sleep(interval_sec)

        print(
            f"\n{Fore.GREEN}Stream complete. "
            f"Alert log: {self.log_path}{Style.RESET_ALL}"
        )

    # ── History ───────────────────────────────────────────────────────────────

    def get_alert_history(self, n: int = 50) -> pd.DataFrame:
        """Return the last *n* alert records as a DataFrame."""
        if not os.path.exists(self.log_path):
            return pd.DataFrame()
        return pd.read_csv(self.log_path).tail(n)


# ══════════════════════════════════════════════════════════════════════════════
# SMSAlertSystem
# ══════════════════════════════════════════════════════════════════════════════

class SMSAlertSystem:
    """
    SMS alert sender supporting Twilio (primary) and
    Fast2SMS (India fallback).  Falls back to mock/log
    mode if no credentials are configured.

    Provider selection
    ------------------
    Set ``"provider"`` in ``config/sms_config.json`` to one of:

    * ``"twilio"``   – international gateway, requires SID + auth-token
    * ``"fast2sms"`` – India DLT-compliant gateway, requires API key
    * ``"mock"``     – (default) prints to console, no real SMS sent

    Rate limiting
    -------------
    * Per-recipient cooldown enforced via the SMS audit log
    * Global ``max_sms_per_hour`` cap (advisory; checked in send loop)
    """

    DEFAULT_CONFIG = {
        "provider": "mock",
        "twilio": {
            "account_sid": "YOUR_TWILIO_SID",
            "auth_token":  "YOUR_TWILIO_TOKEN",
            "from_number": "+1XXXXXXXXXX",
        },
        "fast2sms": {
            "api_key":   "YOUR_FAST2SMS_KEY",
            "sender_id": "FLOODS",
        },
        "recipients": [
            {"name": "AP District Collector",    "phone": "+919XXXXXXXXX"},
            {"name": "SDRF / NDRF Team Lead",    "phone": "+919XXXXXXXXX"},
            {"name": "AP Municipal Authority",   "phone": "+919XXXXXXXXX"},
        ],
        "alert_on_levels":  ["ORANGE", "RED"],
        "cooldown_minutes": 30,
        "max_sms_per_hour": 10,
    }

    def __init__(
        self,
        config_path: str = "config/sms_config.json",
    ) -> None:
        self.config_path = config_path
        self.config      = self._load_config(config_path)
        self.provider    = self.config.get("provider", "mock")
        self.log_path    = "outputs/predictions/sms_log.csv"
        self._init_provider()

    # ── Configuration ─────────────────────────────────────────────────────────

    def _load_config(self, path: str) -> dict:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w") as fh:
                json.dump(self.DEFAULT_CONFIG, fh, indent=2)
            print(f"[SMS] Created default config at {path}")
            print("[SMS] Edit config/sms_config.json with real credentials to go live.")
        with open(path) as fh:
            return json.load(fh)

    def _init_provider(self) -> None:
        if self.provider == "twilio":
            from twilio.rest import Client  # type: ignore
            cfg = self.config["twilio"]
            self.client      = Client(cfg["account_sid"], cfg["auth_token"])
            self.from_number = cfg["from_number"]
            print("[SMS] Twilio provider initialised")
        elif self.provider == "fast2sms":
            self.fast2sms_key = self.config["fast2sms"]["api_key"]
            self.sender_id    = self.config["fast2sms"]["sender_id"]
            print("[SMS] Fast2SMS provider initialised")
        else:
            print("[SMS] Mock mode — messages printed to console, not transmitted")

    # ── Message formatting ────────────────────────────────────────────────────

    def _format_message(self, alert: dict) -> str:
        level     = alert.get("level", "?")
        max_risk  = alert.get("max_risk", 0) * 100
        mean_risk = alert.get("mean_risk", 0) * 100
        area_km2  = alert.get("affected_area_km2", 0)
        ts        = str(alert.get("timestamp", ""))[:19]
        action    = alert.get("message", "")

        # Summarise which rivers are flagged
        raw_zones = str(alert.get("zones_at_risk", ""))
        rivers_hit = sorted({
            e.split("[")[1].rstrip("]").strip()
            for e in raw_zones.split(",")
            if "[" in e
        })
        river_line = ", ".join(rivers_hit) if rivers_hit else "—"

        return (
            f"[AP FLOOD EWS - {level}]\n"
            f"Region: Andhra Pradesh\n"
            f"Rivers at risk: {river_line}\n"
            f"Time: {ts}\n"
            f"Max Risk: {max_risk:.1f}%\n"
            f"Mean Risk: {mean_risk:.1f}%\n"
            f"Affected Area: {area_km2:.1f} km2\n"
            f"Action: {action}\n"
            f"Dashboard: http://127.0.0.1:5050"
        )

    # ── Cooldown / rate-limit ─────────────────────────────────────────────────

    def _check_cooldown(self, phone: str) -> bool:
        """Return True if it is safe to send (recipient not in cooldown)."""
        if not os.path.exists(self.log_path):
            return True

        cooldown_minutes = self.config.get("cooldown_minutes", 30)
        cutoff           = datetime.now() - timedelta(minutes=cooldown_minutes)

        with open(self.log_path, newline="") as fh:
            rows = list(csv.DictReader(fh))

        recent = [
            r for r in rows
            if r.get("phone") == phone
            and datetime.fromisoformat(r["timestamp"]) > cutoff
        ]
        return len(recent) == 0

    def _hourly_count(self) -> int:
        """Count SMS sent in the past 60 minutes (across all recipients)."""
        if not os.path.exists(self.log_path):
            return 0
        cutoff = datetime.now() - timedelta(hours=1)
        with open(self.log_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        return sum(
            1 for r in rows
            if datetime.fromisoformat(r["timestamp"]) > cutoff
            and (r.get("status", "").startswith("sent") or r.get("status", "") == "mock_sent")
        )

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_sms(
        self,
        phone:    str,
        name:     str,
        message:  str,
        status:   str,
        provider: str,
    ) -> None:
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        fields       = ["timestamp", "name", "phone", "provider", "status", "message"]
        write_header = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow({
                "timestamp": datetime.now().isoformat(),
                "name":      name,
                "phone":     phone,
                "provider":  provider,
                "status":    status,
                "message":   message[:80],
            })

    # ── Provider send methods ─────────────────────────────────────────────────

    def send_twilio(self, phone: str, message: str) -> str:
        """
        Send via Twilio REST API.

        Returns the message SID on success.
        Raises ``twilio.base.exceptions.TwilioRestException`` on failure.
        """
        msg = self.client.messages.create(
            body=message,
            from_=self.from_number,
            to=phone,
        )
        if msg.sid is None:
            raise RuntimeError("Twilio message SID missing")
        return msg.sid

    def send_fast2sms(self, phone: str, message: str) -> str:
        """
        Send via Fast2SMS bulk DLT route (India-compliant).

        Strips the ``+91`` prefix that Fast2SMS does not accept.
        Returns the ``request_id`` from the JSON response on success.
        Raises ``requests.HTTPError`` on a non-2xx response.
        """
        import requests  # type: ignore  # available via pip install requests

        number = phone.replace("+91", "").replace("+", "").strip()
        url    = "https://www.fast2sms.com/dev/bulkV2"
        payload = {
            "route":    "q",
            "message":  message,
            "language": "english",
            "flash":    0,
            "numbers":  number,
        }
        headers = {
            "authorization": self.fast2sms_key,
            "Content-Type":  "application/json",
        }
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json().get("request_id", "OK")

    # ── Public dispatch ───────────────────────────────────────────────────────

    def send_alert_sms(self, alert: dict) -> list[dict]:
        """
        Dispatch SMS to all configured recipients for a given alert.

        Guards applied in order:
        1. Alert level must be in ``alert_on_levels``
        2. Global hourly cap must not be exceeded
        3. Per-recipient cooldown must have expired

        Parameters
        ----------
        alert : dict returned by ``FloodAlertSystem.evaluate_risk_map``

        Returns
        -------
        list of result dicts, one per recipient:
        ``{"name": …, "phone": …, "status": …}``
        """
        level = alert.get("level", "GREEN")

        if level not in self.config.get("alert_on_levels", ["ORANGE", "RED"]):
            print(f"[SMS] Level {level} not in alert_on_levels — skipped")
            return []

        max_per_hour = self.config.get("max_sms_per_hour", 10)
        if self._hourly_count() >= max_per_hour:
            print(f"[SMS] Hourly cap ({max_per_hour}) reached — skipped")
            return []

        message  = self._format_message(alert)
        results: list[dict] = []

        for recipient in self.config.get("recipients", []):
            phone = recipient["phone"]
            name  = recipient["name"]

            if not self._check_cooldown(phone):
                print(f"[SMS] Cooldown active for {name} — skipped")
                results.append({"name": name, "status": "cooldown"})
                continue

            try:
                if self.provider == "twilio":
                    ref    = self.send_twilio(phone, message)
                    status = f"sent:{ref}"

                elif self.provider == "fast2sms":
                    ref    = self.send_fast2sms(phone, message)
                    status = f"sent:{ref}"

                else:
                    # ── Mock mode ─────────────────────────────────────────────
                    print(f"\n{'='*52}")
                    print(f"[MOCK SMS]  ➜ {name}  ({phone})")
                    print(f"{'─'*52}")
                    print(message)
                    print(f"{'='*52}")
                    status = "mock_sent"

                self._log_sms(phone, name, message, status, self.provider)
                results.append({"name": name, "phone": phone, "status": status})
                print(f"[SMS] ✓ {name} ({phone}) — {status}")

            except Exception as exc:
                err = str(exc)
                self._log_sms(phone, name, message, f"error:{err}", self.provider)
                results.append({"name": name, "phone": phone, "status": f"error:{err}"})
                print(f"[SMS] ✗ {name}: {err}")

        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_sms_history(self, n: int = 20) -> pd.DataFrame:
        """Return the last *n* SMS log records as a DataFrame."""
        if not os.path.exists(self.log_path):
            return pd.DataFrame()
        return pd.read_csv(self.log_path).tail(n)

    def test_connection(self) -> list[dict]:
        """
        Send a test SMS to the first configured recipient to verify the
        provider credentials and network path.

        Cooldown is bypassed for the duration of this call so it always
        fires regardless of recent history.

        Returns the list of result dicts from ``send_alert_sms``.
        """
        if not self.config.get("recipients"):
            print("[SMS] No recipients configured")
            return []

        test_alert = {
            "level":             "ORANGE",
            "max_risk":          0.72,
            "mean_risk":         0.55,
            "affected_area_km2": 12.5,
            "timestamp":         datetime.now().isoformat(),
            "message":           "TEST — Please ignore. System connectivity check.",
        }

        # Temporarily suppress cooldown
        original_cooldown = self.config.get("cooldown_minutes", 30)
        self.config["cooldown_minutes"] = 0

        # Temporarily restrict to first recipient only
        all_recipients          = self.config["recipients"]
        self.config["recipients"] = all_recipients[:1]

        result = self.send_alert_sms(test_alert)

        # Restore
        self.config["cooldown_minutes"] = original_cooldown
        self.config["recipients"]       = all_recipients

        print(f"[SMS] Test result: {result}")
        return result

    def validate_phone_numbers(self) -> dict[str, bool]:
        """
        Use the ``phonenumbers`` library to validate every recipient number.

        Returns
        -------
        dict mapping ``"name (phone)"`` → ``True/False``
        """
        try:
            import phonenumbers  # type: ignore
        except ImportError:
            print("[SMS] 'phonenumbers' not installed.")
            print("[SMS] Run:  pip install phonenumbers")
            return {}

        results: dict[str, bool] = {}
        for r in self.config.get("recipients", []):
            phone = r["phone"]
            name  = r["name"]
            key   = f"{name} ({phone})"
            try:
                parsed = phonenumbers.parse(phone, None)
                valid  = phonenumbers.is_valid_number(parsed)
                region = phonenumbers.region_code_for_number(parsed)
                results[key] = valid
                marker = "✓" if valid else "✗"
                print(f"[SMS] {marker} {key}  region={region}  valid={valid}")
            except phonenumbers.NumberParseException as exc:
                results[key] = False
                print(f"[SMS] ✗ {key}  parse error: {exc}")

        return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point  ·  python utils/alert_system.py  [--mode ...]
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Andhra Pradesh Flood Early Warning System",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "stream", "sms-test", "validate-phones"],
        default="demo",
        help=(
            "demo             -> 8-step stream hitting all 4 alert tiers (default)\n"
            "stream           -> random-walk stream (--steps N)\n"
            "sms-test         -> send one test SMS to first recipient\n"
            "validate-phones  -> check recipient numbers via phonenumbers"
        ),
    )
    parser.add_argument("--steps",      type=int,   default=8,
                        help="Stream steps         (default: 8)")
    parser.add_argument("--interval",   type=float, default=1.5,
                        help="Seconds between steps (default: 1.5)")
    parser.add_argument("--seed",       type=str,   default="AP_FLOOD",
                        help="RNG seed string      (default: AP_FLOOD)")
    parser.add_argument("--sms-config", type=str,   default="config/sms_config.json",
                        help="SMS config path      (default: config/sms_config.json)")
    args = parser.parse_args()

    print(f"\n{'='*64}")
    print(f"  AP FLOOD EARLY WARNING SYSTEM  --  mode: {args.mode}")
    print(f"{'='*64}\n")

    if args.mode in ("demo", "stream"):
        fas = FloodAlertSystem(
            log_path="outputs/predictions/alert_log.csv",
            demo_mode=(args.mode == "demo"),
        )
        fas.simulate_realtime_stream(
            n_steps=args.steps,
            interval_sec=args.interval,
            seed=args.seed,
            sms_config=args.sms_config,
        )
        print("\n-- Last alerts logged ------------------------------------------")
        df = fas.get_alert_history(args.steps)
        if not df.empty:
            want = ["timestamp", "level", "max_risk", "affected_area_km2", "zones_at_risk"]
            cols = [c for c in want if c in df.columns]
            print(df[cols].to_string(index=False))

    elif args.mode == "sms-test":
        sms = SMSAlertSystem(config_path=args.sms_config)
        print("[SMS] Sending test message to first recipient ...")
        result = sms.test_connection()
        print(f"[SMS] Result: {result}")

    elif args.mode == "validate-phones":
        sms = SMSAlertSystem(config_path=args.sms_config)
        print("[SMS] Validating recipient phone numbers ...")
        sms.validate_phone_numbers()
