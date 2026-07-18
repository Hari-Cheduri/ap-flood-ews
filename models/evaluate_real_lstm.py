import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

from utils.lstm_weather_contract import (
    MODEL_PATH, THRESHOLD_PATH, X_TEST, Y_TEST,
)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run models.train_real_lstm first")

    X = np.load(X_TEST, allow_pickle=False).astype(np.float32)
    y = np.load(Y_TEST, allow_pickle=False).astype(np.int32)
    threshold = float(json.loads(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )["lstm_threshold"])

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    probability = model.predict(X, verbose=0).reshape(-1)
    prediction = (probability >= threshold).astype(np.int32)
    matrix = confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    print("\n==============================")
    print("LSTM TEST EVALUATION")
    print("==============================")
    print("Threshold         :", threshold)
    print("Accuracy          :", accuracy_score(y, prediction))
    print("Balanced accuracy :", balanced_accuracy_score(y, prediction))
    print("Precision         :", precision_score(y, prediction, zero_division=0))
    print("Recall            :", recall_score(y, prediction, zero_division=0))
    print("Specificity       :", tn / (tn + fp) if tn + fp else 0.0)
    print("F1                :", f1_score(y, prediction, zero_division=0))
    print("MCC               :", matthews_corrcoef(y, prediction))
    print("ROC-AUC           :", roc_auc_score(y, probability))
    print("PR-AUC            :", average_precision_score(y, probability))
    print("Confusion matrix:")
    print(matrix)
    print(classification_report(y, prediction, zero_division=0))


if __name__ == "__main__":
    main()
