"""
Live district-level flood-risk map generator.

This version consumes the real 26-district CNN + LSTM hybrid outputs generated
by realtime.build_ap_risk_map_data. It does not load obsolete hybrid encoders
and does not simulate satellite patches or weather sequences.

Input
-----
data/processed/ap_district_risk.json

Outputs
-------
outputs/flood_risk_maps/ap_live_hybrid_risk_map.png
outputs/flood_risk_maps/ap_live_hybrid_risk_map.html
outputs/predictions/ap_live_risk_summary.json

Usage
-----
python -m utils.risk_map_generator

Optional district boundary polygons:
python -m utils.risk_map_generator --boundaries path/to/ap_districts.geojson
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import folium
from folium.plugins import HeatMap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "ap_district_risk.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"

STATE_CENTER = (15.9129, 79.7400)
MAP_BOUNDS = {
    "lat_min": 12.4,
    "lat_max": 19.2,
    "lon_min": 76.7,
    "lon_max": 84.8,
}

LEVELS = {
    "GREEN": {
        "color": "#2ca25f",
        "label": "Green — normal monitoring",
        "priority": 0,
    },
    "YELLOW": {
        "color": "#ffd92f",
        "label": "Yellow — watch",
        "priority": 1,
    },
    "ORANGE": {
        "color": "#f28e2b",
        "label": "Orange — warning",
        "priority": 2,
    },
    "RED": {
        "color": "#d62728",
        "label": "Red — severe warning",
        "priority": 3,
    },
    "UNKNOWN": {
        "color": "#808080",
        "label": "Unavailable",
        "priority": -1,
    },
}


class FloodRiskMapGenerator:
    """Generate live AP district maps from archived hybrid fusion results."""

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
        output_dir: str | Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        self.data_path = Path(data_path)
        self.output_root = Path(output_dir)
        self.risk_dir = self.output_root / "flood_risk_maps"
        self.predictions_dir = self.output_root / "predictions"
        self.risk_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def _load_payload(self) -> Dict[str, Any]:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Missing {self.data_path}. Run "
                "'python -m realtime.build_ap_risk_map_data' first."
            )

        try:
            payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON: {self.data_path}") from exc

        districts = payload.get("districts")
        if not isinstance(districts, list) or not districts:
            raise ValueError("Risk JSON contains no district records")

        return payload

    @staticmethod
    def _normalise_record(record: Dict[str, Any]) -> Dict[str, Any]:
        level = str(record.get("alert_level") or "UNKNOWN").upper()
        if level not in LEVELS:
            level = "UNKNOWN"

        fusion = record.get("fusion_score")
        if not isinstance(fusion, (int, float)):
            fusion = None
        elif not 0.0 <= float(fusion) <= 1.0:
            fusion = max(0.0, min(1.0, float(fusion)))
        else:
            fusion = float(fusion)

        latitude = record.get("latitude")
        longitude = record.get("longitude")
        if not isinstance(latitude, (int, float)):
            raise ValueError(
                f"Missing latitude for {record.get('district_name')}"
            )
        if not isinstance(longitude, (int, float)):
            raise ValueError(
                f"Missing longitude for {record.get('district_name')}"
            )

        clean = dict(record)
        clean["alert_level"] = level
        clean["alert_color"] = LEVELS[level]["color"]
        clean["fusion_score"] = fusion
        clean["risk_percent"] = (
            round(fusion * 100.0, 2) if fusion is not None else None
        )
        clean["latitude"] = float(latitude)
        clean["longitude"] = float(longitude)
        return clean

    def _districts(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            self._normalise_record(record)
            for record in payload["districts"]
        ]

    @staticmethod
    def _popup_html(record: Dict[str, Any]) -> str:
        def pct(value: Any) -> str:
            return (
                f"{float(value) * 100:.1f}%"
                if isinstance(value, (int, float))
                else "Unavailable"
            )

        def number(value: Any, suffix: str = "") -> str:
            return (
                f"{float(value):.2f}{suffix}"
                if isinstance(value, (int, float))
                else "Unavailable"
            )

        return f"""
        <div style="font-family:Arial;min-width:260px">
          <h4 style="margin:0 0 8px 0">{record.get('district_name')}</h4>
          <b>Hybrid alert:</b> {record['alert_level']}<br>
          <b>Fusion score:</b> {pct(record.get('fusion_score'))}<br>
          <hr style="margin:7px 0">
          <b>CNN satellite score:</b> {pct(record.get('cnn_probability'))}<br>
          <b>CNN confidence:</b> {record.get('cnn_confidence') or 'Unavailable'}<br>
          <b>Latest Sentinel scene:</b> {record.get('satellite_latest_scene_utc') or 'Unavailable'}<br>
          <b>Satellite age:</b> {number(record.get('satellite_age_days'), ' days')}<br>
          <b>Satellite freshness:</b> {record.get('satellite_freshness_class') or 'UNKNOWN'}<br>
          <b>CNN effective weight:</b> {pct(record.get('cnn_effective_weight'))}<br>
          <b>LSTM weather score:</b> {pct(record.get('lstm_probability'))}<br>
          <b>LSTM confidence:</b> {record.get('lstm_confidence') or 'Unavailable'}<br>
          <hr style="margin:7px 0">
          <b>Rain, previous 24 h:</b> {number(record.get('rain_last_24h_mm'), ' mm')}<br>
          <b>Rain, previous 72 h:</b> {number(record.get('rain_last_72h_mm'), ' mm')}<br>
          <b>Humidity, previous 24 h:</b> {number(record.get('mean_humidity_last_24h'), '%')}<br>
          <b>Updated:</b> {record.get('pipeline_completed_utc') or 'Unavailable'}<br>
          <p style="margin:8px 0 0 0">{record.get('message') or ''}</p>
        </div>
        """

    @staticmethod
    def _boundary_key(properties: Dict[str, Any]) -> Optional[str]:
        for key in (
            "district_slug",
            "slug",
            "DISTRICT_SLUG",
            "district",
            "DISTRICT",
            "dtname",
            "DTNAME",
            "name",
            "NAME_2",
        ):
            value = properties.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().lower().replace(" ", "_")
        return None

    def _add_boundaries(
        self,
        fmap: folium.Map,
        boundary_path: Path,
        by_slug: Dict[str, Dict[str, Any]],
    ) -> None:
        boundary_payload = json.loads(
            boundary_path.read_text(encoding="utf-8")
        )
        features = boundary_payload.get("features") or []

        for feature in features:
            properties = feature.setdefault("properties", {})
            key = self._boundary_key(properties)
            record = by_slug.get(key or "")
            if record is None:
                continue

            properties.update({
                "district_slug": record.get("district_slug"),
                "district_name": record.get("district_name"),
                "alert_level": record.get("alert_level"),
                "fusion_score": record.get("fusion_score"),
                "risk_percent": record.get("risk_percent"),
                "message": record.get("message"),
                "alert_color": record.get("alert_color"),
            })

        folium.GeoJson(
            boundary_payload,
            name="District risk polygons",
            style_function=lambda feature: {
                "fillColor": feature["properties"].get(
                    "alert_color", LEVELS["UNKNOWN"]["color"]
                ),
                "color": "#222222",
                "weight": 1,
                "fillOpacity": 0.45,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "district_name",
                    "alert_level",
                    "risk_percent",
                    "message",
                ],
                aliases=[
                    "District:",
                    "Alert:",
                    "Fusion risk (%):",
                    "Message:",
                ],
                localize=True,
            ),
            show=True,
        ).add_to(fmap)

    def _save_interactive(
        self,
        records: List[Dict[str, Any]],
        boundary_path: Optional[Path],
    ) -> Path:
        fmap = folium.Map(
            location=list(STATE_CENTER),
            zoom_start=7,
            tiles="CartoDB positron",
            control_scale=True,
        )

        by_slug = {
            str(record.get("district_slug", "")).lower(): record
            for record in records
        }
        if boundary_path is not None:
            if not boundary_path.exists():
                raise FileNotFoundError(
                    f"Boundary GeoJSON not found: {boundary_path}"
                )
            self._add_boundaries(fmap, boundary_path, by_slug)

        marker_layer = folium.FeatureGroup(
            name="District hybrid alerts",
            show=True,
        )
        heat_data = []

        for record in records:
            fusion = record.get("fusion_score")
            score = float(fusion) if isinstance(fusion, (int, float)) else 0.0
            radius = 7.0 + 15.0 * score

            folium.CircleMarker(
                location=[
                    record["latitude"],
                    record["longitude"],
                ],
                radius=radius,
                color="#333333",
                weight=1,
                fill=True,
                fill_color=record["alert_color"],
                fill_opacity=0.85,
                tooltip=(
                    f"{record.get('district_name')}: "
                    f"{record['alert_level']} "
                    f"({score * 100:.1f}%)"
                ),
                popup=folium.Popup(
                    self._popup_html(record),
                    max_width=360,
                ),
            ).add_to(marker_layer)

            if record.get("status") == "success":
                heat_data.append([
                    record["latitude"],
                    record["longitude"],
                    score,
                ])

        marker_layer.add_to(fmap)

        heat_layer = folium.FeatureGroup(
            name="Hybrid fusion heat map",
            show=False,
        )
        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.20,
                max_opacity=0.80,
                radius=28,
                blur=24,
                gradient={
                    0.00: LEVELS["GREEN"]["color"],
                    0.35: LEVELS["YELLOW"]["color"],
                    0.60: LEVELS["ORANGE"]["color"],
                    0.80: LEVELS["RED"]["color"],
                },
            ).add_to(heat_layer)
        heat_layer.add_to(fmap)

        legend_rows = "".join(
            (
                f'<div><span style="display:inline-block;width:13px;'
                f'height:13px;background:{cfg["color"]};margin-right:7px;'
                f'border:1px solid #333"></span>{cfg["label"]}</div>'
            )
            for level, cfg in LEVELS.items()
            if level != "UNKNOWN"
        )
        legend = f"""
        <div style="position:fixed;bottom:30px;left:30px;z-index:9999;
                    background:white;border:1px solid #777;border-radius:6px;
                    padding:12px 15px;font-family:Arial;font-size:12px">
          <b>Live hybrid flood risk</b><br>
          {legend_rows}
          <hr style="margin:7px 0">
          Circle size = fusion score
        </div>
        """
        fmap.get_root().add_child(folium.Element(legend))
        folium.LayerControl(collapsed=False).add_to(fmap)

        path = self.risk_dir / "ap_live_hybrid_risk_map.html"
        fmap.save(str(path))
        return path

    def _save_static(self, records: List[Dict[str, Any]]) -> Path:
        fig, ax = plt.subplots(figsize=(11, 10))
        ax.set_xlim(MAP_BOUNDS["lon_min"], MAP_BOUNDS["lon_max"])
        ax.set_ylim(MAP_BOUNDS["lat_min"], MAP_BOUNDS["lat_max"])
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(
            "Andhra Pradesh — Live District Hybrid Flood Risk"
        )
        ax.grid(True, alpha=0.25)

        for record in records:
            fusion = record.get("fusion_score")
            score = float(fusion) if isinstance(fusion, (int, float)) else 0.0
            ax.scatter(
                record["longitude"],
                record["latitude"],
                s=70 + 650 * score,
                c=record["alert_color"],
                edgecolors="black",
                linewidths=0.7,
                alpha=0.85,
            )

            if record["alert_level"] in {"YELLOW", "ORANGE", "RED"}:
                ax.annotate(
                    str(record.get("district_name")),
                    (record["longitude"], record["latitude"]),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=cfg["color"],
                markeredgecolor="black",
                label=cfg["label"],
                markersize=9,
            )
            for level, cfg in LEVELS.items()
            if level != "UNKNOWN"
        ]
        ax.legend(handles=handles, loc="lower left")

        successful = [
            record for record in records
            if record.get("status") == "success"
        ]
        highest = max(
            successful,
            key=lambda item: (
                LEVELS[item["alert_level"]]["priority"],
                item.get("fusion_score") or 0.0,
            ),
            default=None,
        )
        summary_text = (
            f"Available districts: {len(successful)}/{len(records)}\n"
            + (
                f"Highest alert: {highest['district_name']} — "
                f"{highest['alert_level']} "
                f"({(highest.get('fusion_score') or 0.0) * 100:.1f}%)"
                if highest
                else "No live district results available"
            )
        )
        ax.text(
            0.01,
            0.99,
            summary_text,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
            },
        )

        path = self.risk_dir / "ap_live_hybrid_risk_map.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def _save_summary(
        self,
        payload: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> Path:
        counts = {
            level: sum(
                record["alert_level"] == level
                for record in records
            )
            for level in LEVELS
        }

        ranked = sorted(
            records,
            key=lambda item: (
                LEVELS[item["alert_level"]]["priority"],
                item.get("fusion_score") or -1.0,
            ),
            reverse=True,
        )

        summary = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_generated_utc": payload.get("generated_utc"),
            "source": str(self.data_path),
            "district_count": len(records),
            "successful_count": sum(
                record.get("status") == "success"
                for record in records
            ),
            "alert_counts": counts,
            "top_risk_districts": [
                {
                    "district_slug": record.get("district_slug"),
                    "district_name": record.get("district_name"),
                    "alert_level": record.get("alert_level"),
                    "fusion_score": record.get("fusion_score"),
                    "message": record.get("message"),
                }
                for record in ranked[:10]
            ],
            "limitation": (
                "This is a district-centre hybrid risk map. It is not a "
                "pixel-level inundation or village-level flood-extent map."
            ),
        }

        path = (
            self.predictions_dir
            / "ap_live_risk_summary.json"
        )
        path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return path

    def generate_risk_map(
        self,
        date_str: str | None = None,
        grid_size: int | None = None,
        boundary_geojson: str | Path | None = None,
    ) -> Dict[str, Any]:
        """
        Generate current district-level hybrid maps.

        date_str and grid_size are accepted only for backward compatibility
        with the old synthetic generator. The live data timestamp is read from
        ap_district_risk.json.
        """
        del date_str, grid_size

        payload = self._load_payload()
        records = self._districts(payload)
        boundary_path = (
            Path(boundary_geojson)
            if boundary_geojson is not None
            else None
        )

        png_path = self._save_static(records)
        html_path = self._save_interactive(records, boundary_path)
        summary_path = self._save_summary(payload, records)

        print("\n========================================")
        print("LIVE HYBRID FLOOD-RISK MAP COMPLETE")
        print("========================================")
        print("Districts :", len(records))
        print("PNG       :", png_path.relative_to(PROJECT_ROOT))
        print("HTML      :", html_path.relative_to(PROJECT_ROOT))
        print("Summary   :", summary_path.relative_to(PROJECT_ROOT))

        return {
            "records": records,
            "paths": {
                "png": str(png_path),
                "html": str(html_path),
                "json": str(summary_path),
            },
        }

    def generate_historical_maps(self, *args: Any, **kwargs: Any) -> List[Any]:
        raise RuntimeError(
            "The old method generated fake date-seeded maps. Historical live "
            "maps require saved daily ap_district_risk.json snapshots."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--boundaries",
        default=None,
        help="Optional AP district boundary GeoJSON.",
    )
    args = parser.parse_args()

    generator = FloodRiskMapGenerator(
        data_path=args.data,
        output_dir=args.output_dir,
    )
    generator.generate_risk_map(
        boundary_geojson=args.boundaries,
    )


if __name__ == "__main__":
    main()
