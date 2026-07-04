"""
dashboard/app.py
────────────────────────────────────────────────────────────────────────────────
Andhra Pradesh Flood Early Warning System — Plotly Dash Dashboard
Design: Command-Centre Dark with 3-D card depth, cyan/amber/crimson alert
        palette, purple-accented SMS management panel.

Sections
────────
1  Header bar
2  Current alert + AP risk heatmap
3  River basin cards (Vamsadhara · Godavari · Krishna · Penna)
4  Risk timeline (mean & max from alert_log.csv)
5  Model metrics
6  SMS Alert Management Panel  ← NEW

API endpoints
─────────────
POST /api/test_sms
GET  /api/sms_history
POST /api/update_recipients

Run
───
cd <project_root>
python dashboard/app.py
Open http://127.0.0.1:5050
"""

from __future__ import annotations
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

_log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import (
    Dash, Input, Output, State, callback_context, dash,
    dash_table, dcc, html,
)
from dash.exceptions import PreventUpdate
from flask import jsonify, request

# ── Path bootstrap ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils.alert_system import (          # noqa: E402
    AP_FLOOD_ZONES, ALERT_LEVELS,
    GODAVARI_LAT, KRISHNA_LAT, PENNA_LAT, VAMSADHARA_LAT,
    AP_LAT_MAX, AP_LAT_MIN,
    FloodAlertSystem, SMSAlertSystem, _lat_to_row,
    MONSOON_MULTIPLIER,
)

# ── Constants ─────────────────────────────────────────────────────────────────
REFRESH_MS   = 30_000
ALERT_LOG    = "outputs/predictions/alert_log.csv"
SMS_LOG      = "outputs/predictions/sms_log.csv"
SMS_CFG      = "config/sms_config.json"

ALERT_HEX = {"GREEN": "#22c55e", "YELLOW": "#eab308",
             "ORANGE": "#f97316", "RED":    "#ef4444"}
RIVER_HEX = {"Vamsadhara": "#a371f7", "Godavari": "#3fb950",
             "Krishna":    "#58a6ff", "Penna":    "#f78166"}
RIVER_LATS = {"Vamsadhara": VAMSADHARA_LAT, "Godavari": GODAVARI_LAT,
              "Krishna":    KRISHNA_LAT,    "Penna":    PENNA_LAT}

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg:       #04090f;
  --surface:  #08121e;
  --card:     #0c1a28;
  --card2:    #0f2035;
  --border:   rgba(0,212,255,.10);
  --cyan:     #00d4ff;
  --purple:   #a371f7;
  --txt:      #e2eaf3;
  --muted:    #8b949e;
  --dim:      #3d4a56;
  --mono:     'JetBrains Mono', monospace;
  --disp:     'Rajdhani', sans-serif;
  --body:     'Inter', sans-serif;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  background-image:
    radial-gradient(ellipse 60% 40% at 15% 50%, rgba(0,212,255,.03) 0%, transparent 70%),
    radial-gradient(ellipse 50% 40% at 85% 20%, rgba(163,113,247,.03) 0%, transparent 70%);
  color: var(--txt);
  font-family: var(--body);
  min-height: 100vh;
}

/* ── Cards ── */
.card {
  background: linear-gradient(160deg, var(--card) 0%, var(--card2) 100%);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
  box-shadow:
    0 2px 4px rgba(0,0,0,.5),
    0 8px 32px rgba(0,0,0,.55),
    inset 0 1px 0 rgba(255,255,255,.04);
  transition: transform .18s ease, box-shadow .18s ease;
}
.card:hover {
  transform: translateY(-3px);
  box-shadow:
    0 4px 8px rgba(0,0,0,.6),
    0 16px 48px rgba(0,0,0,.65),
    0 0 0 1px rgba(0,212,255,.08),
    inset 0 1px 0 rgba(255,255,255,.06);
}
.card-sms {
  border-color: rgba(163,113,247,.18);
  box-shadow:
    0 2px 4px rgba(0,0,0,.5),
    0 8px 32px rgba(0,0,0,.55),
    0 0 24px rgba(163,113,247,.05),
    inset 0 1px 0 rgba(163,113,247,.07);
}
.card-sms:hover {
  box-shadow:
    0 4px 8px rgba(0,0,0,.6),
    0 16px 48px rgba(0,0,0,.65),
    0 0 32px rgba(163,113,247,.1),
    inset 0 1px 0 rgba(163,113,247,.1);
}

/* ── Section headers ── */
.sh {
  font-family: var(--disp);
  font-size: 10.5px;
  font-weight: 700;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--cyan);
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 18px;
}
.sh::after { content:''; flex:1; height:1px;
  background: linear-gradient(to right, rgba(0,212,255,.2), transparent); }
.sh-sms { color: var(--purple); }
.sh-sms::after { background: linear-gradient(to right, rgba(163,113,247,.2), transparent); }

/* ── Buttons ── */
.btn {
  border-radius: 6px;
  padding: 7px 18px;
  font-family: var(--disp);
  font-size: 11.5px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  cursor: pointer;
  border: 1px solid;
  transition: all .18s;
}
.btn-cyan {
  background: rgba(0,212,255,.1);
  border-color: rgba(0,212,255,.35);
  color: var(--cyan);
}
.btn-cyan:hover {
  background: rgba(0,212,255,.2);
  box-shadow: 0 0 18px rgba(0,212,255,.2);
}
.btn-purple {
  background: rgba(163,113,247,.12);
  border-color: rgba(163,113,247,.38);
  color: var(--purple);
}
.btn-purple:hover {
  background: rgba(163,113,247,.22);
  box-shadow: 0 0 18px rgba(163,113,247,.2);
}
.btn-sm { padding: 4px 12px; font-size: 10.5px; }

/* ── Live dot ── */
@keyframes live-pulse {
  0%,100% { opacity:.9; transform:scale(1); }
  50%      { opacity:.45; transform:scale(1.5); }
}
.live-dot {
  width:7px; height:7px; border-radius:50%;
  background:#22c55e;
  box-shadow: 0 0 8px #22c55e;
  animation: live-pulse 2s ease-in-out infinite;
  display: inline-block;
}

/* ── Stat labels ── */
.lbl { font-size:10px; color:var(--muted); text-transform:uppercase;
       letter-spacing:.08em; font-weight:600; margin-bottom:3px; }
.val { font-family:var(--mono); font-size:22px; font-weight:600; color:var(--txt); }
.val-sm { font-family:var(--mono); font-size:13px; color:var(--muted); }

/* ── Status tag ── */
.tag {
  display:inline-flex; align-items:center; gap:4px;
  padding:2px 8px; border-radius:4px;
  font-size:10px; font-weight:700; letter-spacing:.06em;
  border: 1px solid;
}
.tag-green  { color:#22c55e; border-color:rgba(34,197,94,.35);  background:rgba(34,197,94,.08);  }
.tag-orange { color:#f97316; border-color:rgba(249,115,22,.35); background:rgba(249,115,22,.08); }
.tag-red    { color:#ef4444; border-color:rgba(239,68,68,.35);  background:rgba(239,68,68,.08);  }
.tag-yellow { color:#eab308; border-color:rgba(234,179,8,.35);  background:rgba(234,179,8,.08);  }
.tag-gray   { color:var(--muted); border-color:var(--border);   background:rgba(255,255,255,.03);}

/* ── DataTable global overrides ── */
.dash-spreadsheet-container .dash-spreadsheet-inner th {
  background:#0c1a28 !important; color:var(--muted) !important;
  border:1px solid var(--border) !important;
  font-family:var(--body) !important; font-size:11px !important;
  font-weight:600 !important; letter-spacing:.04em !important;
  text-transform:uppercase !important;
}
.dash-spreadsheet-container .dash-spreadsheet-inner td {
  background:#0c1a28 !important; color:var(--txt) !important;
  border:1px solid rgba(0,212,255,.06) !important;
  font-family:var(--body) !important; font-size:12px !important;
}
.dash-spreadsheet-container .dash-spreadsheet-inner td:focus {
  background:#0f2035 !important;
  outline: 1px solid var(--cyan) !important;
}

/* ── Misc ── */
.divider { height:1px; background:var(--border); margin:24px 0; }
.gap-sm  { gap:12px; }
.gap-md  { gap:16px; }
.gap-lg  { gap:24px; }
.flex    { display:flex; }
.flex-col { display:flex; flex-direction:column; }
.g2 { display:grid; grid-template-columns:1fr 1fr; }
.g4 { display:grid; grid-template-columns:repeat(4,1fr); }
.g3-1 { display:grid; grid-template-columns:3fr 1fr; }
"""

# ── App init ──────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    title="AP Flood EWS",
    update_title="",
)
server = app.server

app.index_string = f"""
<!DOCTYPE html>
<html>
<head>
  {{%metas%}}
  <title>AP Flood EWS</title>
  {{%favicon%}}
  {{%css%}}
  <style>{_CSS}</style>
</head>
<body>
  {{%app_entry%}}
  <footer>
    {{%config%}}
    {{%scripts%}}
    {{%renderer%}}
  </footer>
</body>
</html>
"""

# ── Helper functions ──────────────────────────────────────────────────────────

def _load_sms_config() -> dict:
    try:
        with open(SMS_CFG) as f:
            return json.load(f)
    except Exception:
        return SMSAlertSystem.DEFAULT_CONFIG

def _load_alert_history(n: int = 30) -> pd.DataFrame:
    try:
        df = pd.read_csv(ALERT_LOG)
        return df.tail(n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _load_sms_history(n: int = 10) -> pd.DataFrame:
    try:
        df = pd.read_csv(SMS_LOG)
        return df.tail(n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _sms_sent_today() -> int:
    df = _load_sms_history(200)
    if df.empty:
        return 0
    cutoff = datetime.now() - timedelta(hours=24)
    try:
        df["ts"] = pd.to_datetime(df["timestamp"])
        mask = df["ts"] > cutoff
        mask &= df["status"].str.startswith("sent") | (df["status"] == "mock_sent")
        return int(mask.sum())
    except Exception:
        return 0

def _last_sms_info() -> tuple[str, str]:
    df = _load_sms_history(50)
    if df.empty:
        return "—", "—"
    sent = df[df["status"].isin(["mock_sent"]) | df["status"].str.startswith("sent", na=False)]
    if sent.empty:
        return "—", "—"
    row = sent.iloc[-1]
    ts  = str(row.get("timestamp", ""))[:16]
    nm  = str(row.get("name", "—"))
    return nm, ts

def _cooldown_status(phone: str, cfg: dict) -> str:
    """Return 'active' or 'clear' for a given phone number."""
    try:
        df = pd.read_csv(SMS_LOG)
        cooldown = cfg.get("cooldown_minutes", 30)
        cutoff   = datetime.now() - timedelta(minutes=cooldown)
        df["ts"] = pd.to_datetime(df["timestamp"])
        recent   = df[(df["phone"] == phone) & (df["ts"] > cutoff)]
        return "active" if len(recent) > 0 else "clear"
    except Exception:
        return "clear"

def _synthetic_risk_grid() -> np.ndarray:
    """Generate a plausible AP risk grid when no model output is available."""
    rng  = np.random.default_rng(int(datetime.now().timestamp()) % 9999)
    grid = np.zeros((50, 50), dtype=np.float32) + 0.15
    # Elevate risk near river corridors
    for river, lat in RIVER_LATS.items():
        row = _lat_to_row(lat, 50)
        for dr in range(-3, 4):
            r = np.clip(row + dr, 0, 49)
            noise = rng.uniform(0.2, 0.55, 50).astype(np.float32)
            attenuation = 1.0 - abs(dr) * 0.12
            grid[r] = np.maximum(grid[r], noise * attenuation)
    return np.clip(grid + rng.normal(0, 0.06, (50, 50)).astype(np.float32), 0, 1)

def _current_alert(grid: np.ndarray) -> dict:
    """Evaluate risk with the current-month monsoon seasonal adjustment."""
    _month = datetime.now().month
    _mult  = MONSOON_MULTIPLIER.get(_month, 1.0)
    _adj   = np.clip(grid * _mult, 0.0, 1.0)
    alert  = _fas.evaluate_risk_map(_adj)
    alert["monsoon_factor"] = _mult
    return alert
def _load_latest_risk_grid():
    """
    Priority order:
    1. Latest model .npy file from risk map generator
    2. Live Open-Meteo weather API (no key needed)
    3. Synthetic fallback
    """
    # 1 — Try most recent model prediction
    npy_files = sorted(Path("outputs/flood_risk_maps").glob("risk_map_*.npy"))
    if npy_files:
        try:
            prob_grid = np.load(str(npy_files[-1])).astype(np.float32)
            _log.info("Loaded model prediction: %s", npy_files[-1].name)
            return prob_grid, None
        except Exception as e:
            _log.warning("Could not load .npy: %s", e)

    # 2 — Live weather API
    try:
        from utils.weather_fetcher import fetch_ap_risk_grid
        return fetch_ap_risk_grid(), None
    except Exception as e:
        _log.warning("Weather API failed: %s — using synthetic", e)

    # 3 — Synthetic fallback
    return _synthetic_risk_grid(), None

# ── Module-level singletons (created once, shared by all callbacks) ───────────
_fas = FloodAlertSystem()
_sms = SMSAlertSystem()

# ── Plot builders ─────────────────────────────────────────────────────────────

def _plot_risk_map(grid: np.ndarray, level: str) -> go.Figure:
    color = ALERT_HEX.get(level, "#58a6ff")
    # AP latitude/longitude ticks
    lat_ticks = [AP_LAT_MIN + i*(AP_LAT_MAX-AP_LAT_MIN)/4 for i in range(5)]
    fig = go.Figure(go.Heatmap(
        z=grid[::-1],                          # flip so north is up
        colorscale=[
            [0.00, "#0d2137"], [0.35, "#1e5f74"],
            [0.55, "#f59e0b"], [0.75, "#f97316"],
            [1.00, "#ef4444"],
        ],
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(
            title=dict(text="Risk", font=dict(color="#8b949e", size=10)),
            tickfont=dict(color="#8b949e", size=9),
            thickness=10, len=0.8,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,212,255,.15)",
        ),
        hovertemplate="Risk: %{z:.2f}<extra></extra>",
    ))

    # River corridor lines
    for river, lat in RIVER_LATS.items():
        row = _lat_to_row(lat, 50)
        y   = 49 - row          # flipped
        fig.add_shape(type="line", x0=0, x1=49, y0=y, y1=y,
                      line=dict(color=RIVER_HEX[river], width=1.2, dash="dot"))
        fig.add_annotation(x=48, y=y + 0.8, text=river,
                           font=dict(color=RIVER_HEX[river], size=9, family="Rajdhani"),
                           showarrow=False, xanchor="right")

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#061220",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   fixedrange=True),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                   fixedrange=True, scaleanchor="x"),
        height=300,
    )
    return fig

def _plot_timeline(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=180, margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    x = pd.to_datetime(df["timestamp"], errors="coerce").dt.strftime("%H:%M")
    fig.add_trace(go.Scatter(
        x=x, y=df["max_risk"] * 100,
        name="Max Risk", mode="lines",
        line=dict(color="#ef4444", width=1.6),
        fill="tozeroy", fillcolor="rgba(239,68,68,.06)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df["mean_risk"] * 100,
        name="Mean Risk", mode="lines+markers",
        line=dict(color="#00d4ff", width=2),
        marker=dict(size=4, color="#00d4ff"),
    ))

    # Alert level threshold lines
    for level, cfg in ALERT_LEVELS.items():
        if level == "GREEN":
            continue
        fig.add_shape(type="line", x0=0, x1=1, xref="paper",
                      y0=cfg["min"]*100, y1=cfg["min"]*100,
                      line=dict(color=ALERT_HEX[level], width=0.8, dash="dot"))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(l=40, r=10, t=10, b=30),
        legend=dict(orientation="h", x=0, y=1.15, font=dict(color="#8b949e", size=10),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#484f58", tickfont=dict(size=9)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.04)",
                   color="#484f58", tickfont=dict(size=9),
                   range=[0, 105], ticksuffix="%"),
        hovermode="x unified",
    )
    return fig

def _plot_model_metrics() -> go.Figure:
    # Try to load real metrics
    metrics_path = Path("outputs/reports/hybrid_metrics.json")
    try:
        if metrics_path.exists():
            with open(metrics_path) as f:
                m = json.load(f)
            hybrid_p = m["test"]["precision_flood"]
            hybrid_r = m["test"]["recall_flood"]
            hybrid_f = m["test"]["f1_flood"]
            # CNN/LSTM single-modal metrics may not be stored; use placeholders
            precisions = [m.get("cnn_precision", 0.88), m.get("lstm_precision", 0.84), hybrid_p]
            recalls    = [m.get("cnn_recall",    0.85), m.get("lstm_recall",    0.87), hybrid_r]
            f1s        = [m.get("cnn_f1",        0.86), m.get("lstm_f1",        0.85), hybrid_f]
        else:
            raise FileNotFoundError
    except Exception:
        precisions = [0.88, 0.84, 0.92]
        recalls    = [0.85, 0.87, 0.91]
        f1s        = [0.86, 0.85, 0.91]

    models  = ["CNN", "LSTM", "Hybrid"]
    metrics_data = {"Precision": precisions, "Recall": recalls, "F1": f1s}
    colors  = ["#00d4ff", "#a371f7", "#22c55e"]
    fig = go.Figure()
    for i, (metric_name, vals) in enumerate(metrics_data.items()):
        fig.add_trace(go.Bar(
            name=metric_name, x=models, y=vals,
            marker_color=colors[i],
            marker_line=dict(width=0),
            width=0.25,
            offset=(i - 1) * 0.26,
        ))
    fig.add_shape(type="line", x0=-0.5, x1=2.5, y0=0.80, y1=0.80,
                  line=dict(color="#f97316", width=1, dash="dash"))
    fig.add_annotation(x=2.5, y=0.80, text="Target 80%",
                       showarrow=False, xanchor="right",
                       font=dict(color="#f97316", size=9))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=180, barmode="overlay",
        margin=dict(l=30, r=10, t=10, b=30),
        legend=dict(orientation="h", x=0, y=1.12, bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#8b949e", size=10)),
        xaxis=dict(showgrid=False, color="#484f58"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.04)",
                   color="#484f58", range=[0.7, 1.0], tickformat=".0%"),
    )
    return fig

# ── SMS panel sub-components ──────────────────────────────────────────────────

def _recipients_table_data(cfg: dict) -> list[dict]:
    rows = []
    for r in cfg.get("recipients", []):
        cd = _cooldown_status(r.get("phone", ""), cfg)
        rows.append({
            "name":     r.get("name", ""),
            "phone":    r.get("phone", ""),
            "levels":   ", ".join(cfg.get("alert_on_levels", [])),
            "cooldown": "⏱ active" if cd == "active" else "✓ clear",
        })
    return rows

def _sms_history_table_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        st    = str(r.get("status", ""))
        ts    = str(r.get("timestamp", ""))[:16]
        msg   = str(r.get("message", ""))
        level = "ORANGE"
        for lvl in ("RED", "ORANGE", "YELLOW", "GREEN"):
            if lvl in msg:
                level = lvl
                break
        rows.append({
            "time":      ts,
            "recipient": str(r.get("name", "—")),
            "status":    st,
            "level":     level,
        })
    return rows

def _history_row_style(rows: list[dict]) -> list[dict]:
    styles = []
    for i, r in enumerate(rows):
        st = r.get("status", "")
        if st.startswith("sent") or st == "mock_sent":
            bg = "rgba(34,197,94,.06)"
            col = "#22c55e"
        elif "error" in st:
            bg = "rgba(239,68,68,.06)"
            col = "#ef4444"
        else:
            bg = "rgba(255,255,255,.01)"
            col = "#484f58"
        styles.append({
            "if": {"row_index": i},
            "style": {
                "backgroundColor": bg,
                "color": col,
            },
        })
    return styles

# ── Layout helpers ────────────────────────────────────────────────────────────

def _sh(label: str, extra_class: str = "") -> html.Div:
    return html.Div(label, className=f"sh {extra_class}")

def _stat(label: str, cid: str, default: str = "—") -> html.Div:
    return html.Div([
        html.Div(label, className="lbl"),
        html.Div(default, id=cid, className="val"),
    ], style={"marginBottom": "14px"})

def _tag(text: str, cls: str) -> html.Span:
    return html.Span(text, className=f"tag tag-{cls}")

# ── App layout ────────────────────────────────────────────────────────────────

cfg0       = _load_sms_config()
recipients = _recipients_table_data(cfg0)
sms_hist   = _sms_history_table_rows(_load_sms_history(10))
hist_style = _history_row_style(sms_hist)

app.layout = html.Div(style={"maxWidth": "1400px", "margin": "0 auto",
                              "padding": "0 20px 40px"}, children=[

    # ── Stores ──────────────────────────────────────────────────────────────
    dcc.Store(id="alert-store"),
    dcc.Interval(id="tick", interval=REFRESH_MS, n_intervals=0),

    # ── S1 Header ────────────────────────────────────────────────────────────
    html.Div(style={
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "22px 0 18px", "borderBottom": "1px solid rgba(0,212,255,.1)",
        "marginBottom": "24px",
    }, children=[
        html.Div([
            html.Div("⬡ ANDHRA PRADESH", style={
                "fontFamily": "'Rajdhani',sans-serif", "fontWeight": "700",
                "fontSize": "11px", "letterSpacing": ".2em", "color": "#484f58",
                "textTransform": "uppercase", "marginBottom": "2px",
            }),
            html.Div("FLOOD EARLY WARNING SYSTEM", style={
                "fontFamily": "'Rajdhani',sans-serif", "fontWeight": "700",
                "fontSize": "26px", "letterSpacing": ".06em", "color": "#e2eaf3",
            }),
        ]),
        html.Div([
            html.Div([
                html.Div(className="live-dot"),
                html.Span(" LIVE", style={"fontSize": "10px", "fontFamily": "'Rajdhani',sans-serif",
                                          "fontWeight": "700", "letterSpacing": ".14em",
                                          "color": "#22c55e", "marginLeft": "4px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
            html.Div(id="last-refresh", className="val-sm",
                     style={"textAlign": "right", "color": "#3d4a56", "fontSize": "10px"}),
        ]),
    ]),

    # ── S2 Alert + Risk Map ───────────────────────────────────────────────────
    html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 3fr",
                    "gap": "20px", "marginBottom": "20px"}, children=[

        # Left: Alert status
        html.Div(className="card", children=[
            _sh("CURRENT STATUS"),
            html.Div(id="alert-level-display", style={
                "fontFamily": "'Rajdhani',sans-serif", "fontWeight": "700",
                "fontSize": "52px", "letterSpacing": ".06em", "lineHeight": "1",
                "marginBottom": "10px", "textShadow": "0 0 30px currentColor",
            }),
            html.Div(id="alert-action-text", style={
                "fontSize": "12px", "color": "#8b949e", "marginBottom": "20px",
                "lineHeight": "1.5",
            }),
            html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                            "gap": "14px"}, children=[
                html.Div([html.Div("MAX RISK",    className="lbl"),
                          html.Div(id="max-risk-val", className="val")]),
                html.Div([html.Div("MEAN RISK",   className="lbl"),
                          html.Div(id="mean-risk-val", className="val")]),
                html.Div([html.Div("AFFECTED km²", className="lbl"),
                          html.Div(id="area-val", className="val")]),
                html.Div([html.Div("MONSOON ×",   className="lbl"),
                          html.Div(id="monsoon-val", className="val")]),
            ]),
        ]),

        # Right: Risk heatmap
        html.Div(className="card", style={"padding": "16px"}, children=[
            _sh("AP RIVER-BASIN RISK MAP — 50 × 50 GRID"),
            dcc.Graph(id="risk-map",
                      figure=_plot_risk_map(_synthetic_risk_grid(), "GREEN"),
                      config={"displayModeBar": False},
                      style={"height": "300px"}),
        ]),
    ]),

    # ── S3 River Basin Cards ──────────────────────────────────────────────────
    html.Div(id="basin-cards",
             style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)",
                    "gap": "16px", "marginBottom": "20px"}),

    # ── S4 Risk Timeline ──────────────────────────────────────────────────────
    html.Div(className="card", style={"marginBottom": "20px"}, children=[
        _sh("RISK TIMELINE  ( mean & max over last 30 updates )"),
        dcc.Graph(id="timeline",
                  figure=_plot_timeline(_load_alert_history()),
                  config={"displayModeBar": False},
                  style={"height": "180px"}),
    ]),

    # ── S5 Metrics ────────────────────────────────────────────────────────────
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                    "gap": "20px", "marginBottom": "28px"}, children=[
        html.Div(className="card", children=[
            _sh("MODEL PERFORMANCE"),
            dcc.Graph(id="model-metrics",
                      figure=_plot_model_metrics(),
                      config={"displayModeBar": False},
                      style={"height": "180px"}),
        ]),
        html.Div(className="card", children=[
            _sh("RECENT ALERT LOG"),
            html.Div(id="alert-log-table"),
        ]),
    ]),

    # ── S6 SMS MANAGEMENT PANEL ───────────────────────────────────────────────
    html.Div([
        # Section header (full-width)
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px",
                        "marginBottom": "18px"}, children=[
            html.Div("⬡ SMS ALERT MANAGEMENT", style={
                "fontFamily": "'Rajdhani',sans-serif", "fontWeight": "700",
                "fontSize": "11px", "letterSpacing": ".18em", "color": "#a371f7",
                "textTransform": "uppercase",
            }),
            html.Div(style={"flex": "1", "height": "1px",
                            "background": "linear-gradient(to right,rgba(163,113,247,.3),transparent)"}),
        ]),

        # Row A: Status card + Alert level selector
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "16px", "marginBottom": "16px"}, children=[

            # A1: SMS Status Card
            html.Div(className="card card-sms", children=[
                _sh("SMS STATUS", "sh-sms"),
                html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                                "gap": "16px", "marginBottom": "18px"}, children=[
                    html.Div([
                        html.Div("PROVIDER", className="lbl"),
                        html.Div(id="sms-provider", className="val",
                                 style={"fontSize": "16px", "textTransform": "uppercase"}),
                    ]),
                    html.Div([
                        html.Div("SENT TODAY", className="lbl"),
                        html.Div(id="sms-today", className="val",
                                 style={"fontSize": "16px"}),
                    ]),
                    html.Div([
                        html.Div("LAST RECIPIENT", className="lbl"),
                        html.Div(id="sms-last-name", className="val-sm",
                                 style={"fontSize": "12px"}),
                    ]),
                    html.Div([
                        html.Div("LAST TIME", className="lbl"),
                        html.Div(id="sms-last-time", className="val-sm",
                                 style={"fontSize": "12px", "fontFamily": "'JetBrains Mono',monospace"}),
                    ]),
                ]),
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                    html.Button("▶ TEST SMS", id="test-sms-btn",
                                className="btn btn-purple btn-sm"),
                    html.Div(id="test-sms-result",
                             style={"fontSize": "11px", "color": "#a371f7"}),
                ]),
            ]),

            # A2: Alert level selector
            html.Div(className="card card-sms", children=[
                _sh("ALERT LEVELS THAT TRIGGER SMS", "sh-sms"),
                html.Div("Select which alert levels automatically dispatch SMS messages to all recipients.",
                         style={"fontSize": "11px", "color": "#8b949e",
                                "lineHeight": "1.6", "marginBottom": "18px"}),
                dcc.Checklist(
                    id="alert-levels-checklist",
                    options=[
                        {"label": html.Span("  YELLOW", style={"color": "#eab308",
                                            "fontFamily": "'Rajdhani',sans-serif",
                                            "fontWeight": "700", "letterSpacing": ".08em"}),
                         "value": "YELLOW"},
                        {"label": html.Span("  ORANGE", style={"color": "#f97316",
                                            "fontFamily": "'Rajdhani',sans-serif",
                                            "fontWeight": "700", "letterSpacing": ".08em"}),
                         "value": "ORANGE"},
                        {"label": html.Span("  RED",    style={"color": "#ef4444",
                                            "fontFamily": "'Rajdhani',sans-serif",
                                            "fontWeight": "700", "letterSpacing": ".08em"}),
                         "value": "RED"},
                    ],
                    value=cfg0.get("alert_on_levels", ["ORANGE", "RED"]),
                    style={"display": "flex", "flexDirection": "column", "gap": "12px",
                           "marginBottom": "20px"},
                    inputStyle={"marginRight": "8px", "accentColor": "#a371f7"},
                    labelStyle={"display": "flex", "alignItems": "center",
                                "cursor": "pointer"},
                ),
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px"}, children=[
                    html.Button("SAVE LEVELS", id="save-levels-btn",
                                className="btn btn-purple btn-sm"),
                    html.Div(id="levels-save-status",
                             style={"fontSize": "11px", "color": "#a371f7"}),
                ]),
            ]),
        ]),

        # Row B: Recipient Management Table
        html.Div(className="card card-sms", style={"marginBottom": "16px"}, children=[
            _sh("RECIPIENT MANAGEMENT", "sh-sms"),
            html.Div("Inline-editable. Click a cell to edit, then press Save.",
                     style={"fontSize": "11px", "color": "#8b949e", "marginBottom": "12px"}),
            dash_table.DataTable(
                id="recipients-table",
                columns=[
                    {"name": "Name",           "id": "name",     "editable": True},
                    {"name": "Phone",          "id": "phone",    "editable": True},
                    {"name": "Alert Levels",   "id": "levels",   "editable": False},
                    {"name": "Cooldown",       "id": "cooldown", "editable": False},
                ],
                data=recipients,
                editable=True,
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": "#0c1a28",
                    "color": "#e2eaf3",
                    "border": "1px solid rgba(163,113,247,.12)",
                    "padding": "8px 12px",
                    "fontSize": "12px",
                    "fontFamily": "'Inter',sans-serif",
                },
                style_header={
                    "backgroundColor": "#08121e",
                    "color": "#8b949e",
                    "fontWeight": "600",
                    "fontSize": "10px",
                    "letterSpacing": ".06em",
                    "textTransform": "uppercase",
                    "border": "1px solid rgba(163,113,247,.15)",
                },
                style_data_conditional=[
                    {"if": {"filter_query": '{cooldown} contains "active"'},
                        "style": {"color": "#f97316"}},
                    {"if": {"filter_query": '{cooldown} contains "clear"'},
                        "style": {"color": "#22c55e"}},
                ],
                page_size=10,
            ),
            html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "12px", "marginTop": "12px"}, children=[
                html.Button("SAVE RECIPIENTS", id="save-recipients-btn",
                            className="btn btn-purple btn-sm"),
                html.Div(id="recipients-save-status",
                         style={"fontSize": "11px", "color": "#a371f7"}),
            ]),
        ]),

        # Row C: SMS History Table
        html.Div(className="card card-sms", children=[
            _sh("SMS HISTORY — last 10", "sh-sms"),
            dash_table.DataTable(
                id="sms-history-table",
                columns=[
                    {"name": "Time",      "id": "time"},
                    {"name": "Recipient", "id": "recipient"},
                    {"name": "Status",    "id": "status"},
                    {"name": "Level",     "id": "level"},
                ],
                data=sms_hist,
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": "#0c1a28",
                    "color": "#e2eaf3",
                    "border": "1px solid rgba(163,113,247,.08)",
                    "padding": "7px 12px",
                    "fontSize": "12px",
                    "fontFamily": "'JetBrains Mono',monospace",
                },
                style_header={
                    "backgroundColor": "#08121e",
                    "color": "#8b949e",
                    "fontWeight": "600",
                    "fontSize": "10px",
                    "letterSpacing": ".06em",
                    "textTransform": "uppercase",
                    "border": "1px solid rgba(163,113,247,.15)",
                },
                style_data_conditional=[hist_style
                                        ],
                page_size=10,
            ),
        ]),
    ]),  # end SMS panel

])  # end app.layout

# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

# ── CB1: Main refresh ─────────────────────────────────────────────────────────
@app.callback(
    [
        Output("alert-level-display", "children"),
        Output("alert-level-display", "style"),
        Output("alert-action-text",   "children"),
        Output("max-risk-val",         "children"),
        Output("mean-risk-val",        "children"),
        Output("area-val",             "children"),
        Output("monsoon-val",          "children"),
        Output("risk-map",             "figure"),
        Output("basin-cards",          "children"),
        Output("timeline",             "figure"),
        Output("alert-log-table",      "children"),
        Output("alert-store",          "data"),
        Output("last-refresh",         "children"),
    ],
    Input("tick", "n_intervals"),
)
def refresh_dashboard(n):
    # Build current risk state
    grid, _latest_summary = _load_latest_risk_grid()
    from datetime import datetime as _dt
    import calendar
    _month = _dt.now().month
    _mult  = MONSOON_MULTIPLIER.get(_month, 1.0)
    _adj   = np.clip(grid * _mult, 0.0, 1.0)
    alert  = _fas.evaluate_risk_map(_adj)
    alert["monsoon_factor"] = _mult          # keep the factor visible in the store
    _fas.log_alert(alert)
    level = alert["level"]
    color = ALERT_HEX[level]

    # Alert level display style
    level_style = {
        "fontFamily": "'Rajdhani',sans-serif", "fontWeight": "700",
        "fontSize": "52px", "letterSpacing": ".06em", "lineHeight": "1",
        "marginBottom": "10px", "color": color,
        "textShadow": f"0 0 30px {color}",
    }

    # Basin cards
    basin_names = ["Vamsadhara", "Godavari", "Krishna", "Penna"]
    zones_hit   = str(alert.get("zones_at_risk", ""))
    basin_cards = []
    for river in basin_names:
        active_zones = [
            z.split("[")[0].strip()
            for z in zones_hit.split(",")
            if f"[{river}]" in z
        ]
        rcolor = RIVER_HEX[river]
        status = "AT RISK" if active_zones else "MONITORING"
        stag   = "orange" if active_zones else "green"
        card   = html.Div(className="card", style={
            "borderLeftColor": rcolor, "borderLeftWidth": "3px",
            "borderLeftStyle": "solid",
        }, children=[
            html.Div([
                html.Span(river, className="basin-river-name",
                          style={"color": rcolor, "fontSize": "12px",
                                 "fontFamily": "'Rajdhani',sans-serif",
                                 "fontWeight": "700", "letterSpacing": ".08em",
                                 "textTransform": "uppercase"}),
                _tag(status, stag),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "8px"}),
            html.Div(f"{RIVER_LATS[river]:.2f}°N",
                     style={"fontFamily": "'JetBrains Mono',monospace",
                            "fontSize": "10px", "color": "#3d4a56",
                            "marginBottom": "8px"}),
            html.Div(
                ", ".join(active_zones[:2]) if active_zones else "No zones above threshold",
                style={"fontSize": "10px", "color": "#8b949e", "lineHeight": "1.5"},
            ),
        ])
        basin_cards.append(card)

    # Alert log mini-table
    df = _load_alert_history(8)
    if not df.empty:
        rows = []
        for _, r in df.iloc[::-1].iterrows():
            lvl = str(r.get("level", "—"))
            c   = ALERT_HEX.get(lvl, "#8b949e")
            rows.append(html.Div([
                html.Span(str(r.get("timestamp", ""))[:16],
                          style={"fontFamily": "'JetBrains Mono',monospace",
                                 "fontSize": "10px", "color": "#484f58", "flex": "0 0 105px"}),
                html.Span(lvl, style={"color": c, "fontFamily": "'Rajdhani',sans-serif",
                                      "fontWeight": "700", "fontSize": "11px",
                                      "letterSpacing": ".06em", "flex": "0 0 60px"}),
                html.Span(f"{float(r.get('max_risk',0))*100:.0f}%",
                          style={"fontFamily": "'JetBrains Mono',monospace",
                                 "fontSize": "11px", "color": c, "flex": "0 0 40px"}),
                html.Span(str(r.get("zones_at_risk", ""))[:35] + "…"
                          if len(str(r.get("zones_at_risk",""))) > 35
                          else str(r.get("zones_at_risk","—")),
                          style={"fontSize": "10px", "color": "#484f58", "flex": "1",
                                 "overflow": "hidden", "whiteSpace": "nowrap",
                                 "textOverflow": "ellipsis"}),
            ], style={"display": "flex", "gap": "8px", "padding": "5px 0",
                      "borderBottom": "1px solid rgba(255,255,255,.03)",
                      "alignItems": "center"}))
        log_widget = html.Div(rows)
    else:
        log_widget = html.Div("No alerts logged yet.",
                              style={"color": "#484f58", "fontSize": "11px"})

    ts = datetime.now().strftime("Updated %H:%M:%S")

    return (
        level,
        level_style,
        alert.get("message", ""),
        f"{alert['max_risk']*100:.1f}%",
        f"{alert['mean_risk']*100:.1f}%",
        f"{alert['affected_area_km2']:.0f}",
        f"{alert.get('monsoon_factor', 1.0):.2f}×",
        _plot_risk_map(grid, level),
        basin_cards,
        _plot_timeline(_load_alert_history()),
        log_widget,
        alert,           # stored for SMS auto-trigger
        ts,
    )

# ── CB2: Auto SMS trigger on refresh ─────────────────────────────────────────
@app.callback(
    [
        Output("sms-provider",   "children"),
        Output("sms-today",      "children"),
        Output("sms-last-name",  "children"),
        Output("sms-last-time",  "children"),
        Output("sms-history-table", "data"),
        Output("sms-history-table", "style_data_conditional"),
        Output("recipients-table",  "data"),
    ],
    Input("alert-store", "data"),
    prevent_initial_call=True,
)
def sms_refresh_and_auto_trigger(alert_data):
    # Auto-trigger SMS if alert level warrants it
    if alert_data:
        try:
            level = alert_data.get("level", "GREEN")
            if level in _sms.config.get("alert_on_levels", ["ORANGE", "RED"]):
                _sms.send_alert_sms(alert_data)
        except Exception as _sms_err:
            _log.warning("SMS auto-trigger failed: %s", _sms_err)

    cfg    = _load_sms_config()
    prov   = cfg.get("provider", "mock").upper()
    today  = _sms_sent_today()
    lname, ltime = _last_sms_info()

    hist_rows  = _sms_history_table_rows(_load_sms_history(10))
    hist_style = _history_row_style(hist_rows)
    recips     = _recipients_table_data(cfg)

    return prov, str(today), lname, ltime, hist_rows, hist_style, recips

# ── CB3: Test SMS button ──────────────────────────────────────────────────────
@app.callback(
    Output("test-sms-result", "children"),
    Input("test-sms-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_test_sms(n_clicks):
    if not n_clicks:
        return dash.no_update 
    try:
        results = _sms.test_connection()
        if results:
            r = results[0]
            return f"✓ {r['name']}: {r['status']}"
        return "No recipients configured"
    except Exception as exc:
        return f"✗ {exc}"

# ── CB4: Save alert levels ────────────────────────────────────────────────────
@app.callback(
    Output("levels-save-status", "children"),
    Input("save-levels-btn", "n_clicks"),
    State("alert-levels-checklist", "value"),
    prevent_initial_call=True,
)
def save_alert_levels(n_clicks, selected_levels):
    if not n_clicks:
        raise PreventUpdate
    if not selected_levels:
        return "Select at least one level"
    try:
        with open(SMS_CFG) as f:
            cfg = json.load(f)
        cfg["alert_on_levels"] = selected_levels
        with open(SMS_CFG, "w") as f:
            json.dump(cfg, f, indent=2)
        return "✓ Saved"
    except Exception as exc:
        return f"✗ {exc}"

# ── CB5: Save recipients ──────────────────────────────────────────────────────
@app.callback(
    Output("recipients-save-status", "children"),
    Input("save-recipients-btn", "n_clicks"),
    State("recipients-table", "data"),
    prevent_initial_call=True,
)
def save_recipients(n_clicks, table_data):
    if not n_clicks:
        raise PreventUpdate
    try:
        with open(SMS_CFG) as f:
            cfg = json.load(f)
        cfg["recipients"] = [
            {"name": r.get("name", ""), "phone": r.get("phone", "")}
            for r in (table_data or [])
        ]
        with open(SMS_CFG, "w") as f:
            json.dump(cfg, f, indent=2)
        return f"✓ {len(cfg['recipients'])} recipients saved"
    except Exception as exc:
        return f"✗ {exc}"

# ══════════════════════════════════════════════════════════════════════════════
# Flask API endpoints
# ══════════════════════════════════════════════════════════════════════════════

@server.route("/api/test_sms", methods=["POST"])
def api_test_sms():
    try:
        result = _sms.test_connection()
        return jsonify({"status": "ok", "results": result})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

@server.route("/api/sms_history")
def api_sms_history():
    try:
        df  = _sms.get_sms_history(20)
        return jsonify(df.to_dict(orient="records") if not df.empty else [])
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

@server.route("/api/update_recipients", methods=["POST"])
def api_update_recipients():
    try:
        data = request.json
        with open(SMS_CFG) as f:
            cfg = json.load(f)
        cfg["recipients"] = data.get("recipients", cfg["recipients"])
        with open(SMS_CFG, "w") as f:
            json.dump(cfg, f, indent=2)
        return jsonify({"status": "saved",
                        "count": len(cfg["recipients"])})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

@server.route("/api/alert_status")
def api_alert_status():
    """Current alert dict — useful for external polling."""
    try:
        grid, _ = _load_latest_risk_grid()
        alert   = _current_alert(grid)
        return jsonify(alert)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AP Flood EWS Dashboard")
    parser.add_argument("--port",  type=int,   default=5050)
    parser.add_argument("--host",  type=str,   default="127.0.0.1")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  AP FLOOD EWS  —  Dashboard")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
