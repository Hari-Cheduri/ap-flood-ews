"""
main.py
--------
CLI entrypoint for the Flood Monitoring System.

Commands
--------
    python main.py train    – train or fine-tune the hybrid model
    python main.py predict  – run batch inference on processed data
    python main.py serve    – launch the Flask+Dash dashboard
    python main.py pipeline – end-to-end: ingest → preprocess → predict → alert

Usage
-----
    python main.py --help
    python main.py train   --epochs 50 --batch-size 8
    python main.py predict --station G-04
    python main.py serve   --port 8050
"""

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/reports/system.log", mode="a"),
    ],
)
logger = logging.getLogger("flood_monitor")


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_train(args):
    """Train the hybrid CNN+LSTM model."""
    logger.info("=== TRAINING MODE ===")
    logger.info("epochs=%d  batch_size=%d  lr=%g",
                args.epochs, args.batch_size, args.lr)

    from utils.data_loader  import DataLoader
    from utils.preprocessor import Preprocessor
    from models.hybrid_model import HybridCNNLSTM
    import numpy as np

    loader = DataLoader()
    prep   = Preprocessor()

    # ── Sensor data ──────────────────────────────────────────────────────
    try:
        df = loader.load_all_sensors()
        df = prep.resample_sensor_data(df, freq="1H")
        df = prep.smooth_sensor_data(df, window=3)

        X_ts, y_risk = prep.create_sequences(df, window_size=72, n_ahead=24)
        logger.info("Sensor sequences: X=%s  y=%s", X_ts.shape, y_risk.shape)

        X_tr, X_val, X_te, y_tr, y_val, y_te = prep.split_data(X_ts, y_risk)
    except FileNotFoundError:
        logger.warning("No sensor CSV found – using synthetic data for demonstration.")
        X_tr  = np.random.rand(200, 72, 6).astype("float32")
        y_tr  = (np.random.rand(200, 24) > 0.8).astype("float32")
        X_val = np.random.rand(40,  72, 6).astype("float32")
        y_val = (np.random.rand(40,  24) > 0.8).astype("float32")

    # ── Synthetic image data (replace with real GeoTIFF loader) ──────────
    img_shape = (256, 256, 7)
    X_img_tr  = np.random.rand(len(X_tr),  *img_shape).astype("float32")
    X_img_val = np.random.rand(len(X_val), *img_shape).astype("float32")
    y_maps_tr  = np.random.rand(len(X_tr),  256, 256, 1).astype("float32")
    y_maps_val = np.random.rand(len(X_val), 256, 256, 1).astype("float32")
    y_risk_tr  = y_tr[:, 0:1]
    y_risk_val = y_val[:, 0:1]

    # ── Model ─────────────────────────────────────────────────────────────
    model = HybridCNNLSTM(image_shape=img_shape)
    model.compile(learning_rate=args.lr)
    model.summary()

    history = model.fit(
        X_img_tr, X_tr, y_risk_tr, y_maps_tr,
        X_img_val, X_val, y_risk_val, y_maps_val,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    model.save("outputs/predictions/hybrid_final.h5")
    logger.info("=== TRAINING COMPLETE ===")


def cmd_predict(args):
    """Run inference for a given station using the latest model."""
    logger.info("=== PREDICTION MODE  station=%s ===", args.station)

    from models.hybrid_model import HybridCNNLSTM
    from utils.alert_system  import AlertSystem
    import numpy as np

    model_path = Path("outputs/predictions/hybrid_final.h5")
    if not model_path.exists():
        logger.error("No trained model found at %s. Run `python main.py train` first.", model_path)
        sys.exit(1)

    model  = HybridCNNLSTM.load(str(model_path))
    alerts = AlertSystem()

    # Synthetic inference inputs – replace with real data pipeline
    image         = np.random.rand(256, 256, 7).astype("float32")
    sensor_window = np.random.rand(72, 6).astype("float32")

    result = model.predict(image, sensor_window)
    logger.info("Prediction: risk=%.4f  level=%s  flooded=%.1f%%",
                result["risk_score"], result["alert_level"], result["flooded_area_pct"])

    if result["alert_level"] in ("HIGH", "CRITICAL"):
        alert = alerts.create_alert(
            station_id=args.station,
            risk_score=result["risk_score"],
            flooded_area_pct=result["flooded_area_pct"],
        )
        alerts.dispatch(alert, send_email=False, send_webhook=False)

    return result


def cmd_serve(args):
    """Launch the Flask + Dash dashboard server."""
    logger.info("=== SERVING DASHBOARD on port %d ===", args.port)
    from dashboard.app import server
    server.run(host="0.0.0.0", port=args.port, debug=args.debug)


def cmd_pipeline(args):
    """Full end-to-end pipeline: ingest → preprocess → predict → alert."""
    logger.info("=== END-TO-END PIPELINE ===")
    cmd_predict(args)       # prediction & alerting
    logger.info("Pipeline complete. Check outputs/reports/ for logs.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flood_monitor",
        description="Hybrid CNN+LSTM Real-Time Flood Monitoring System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the hybrid model")
    p_train.add_argument("--epochs",     type=int,   default=50)
    p_train.add_argument("--batch-size", type=int,   default=8)
    p_train.add_argument("--lr",         type=float, default=5e-4)

    # predict
    p_pred = sub.add_parser("predict", help="Run batch inference")
    p_pred.add_argument("--station", type=str, default="G-01")

    # serve
    p_serve = sub.add_parser("serve", help="Launch dashboard")
    p_serve.add_argument("--port",  type=int,  default=8050)
    p_serve.add_argument("--debug", action="store_true")

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Run full end-to-end pipeline")
    p_pipe.add_argument("--station", type=str, default="G-01")

    return parser


def main():
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "train":    cmd_train,
        "predict":  cmd_predict,
        "serve":    cmd_serve,
        "pipeline": cmd_pipeline,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
