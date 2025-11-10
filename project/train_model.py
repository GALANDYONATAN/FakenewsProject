# train_model.py
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text as sql_text

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)
import joblib

from sentence_transformers import SentenceTransformer
import sklearn
import sentence_transformers as st_pkg



DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})

# CLAIM: video_transcription_text + claim_status
df = pd.read_sql(
    """
    SELECT video_transcription_text, claim_status
    FROM tiktok_csv
    WHERE video_transcription_text IS NOT NULL
      AND claim_status IS NOT NULL
    """,
    engine
).dropna()

y = df["claim_status"].replace({"claim": 1, "opinion": 0}).astype(int)
X = df["video_transcription_text"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)



class SentenceEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2", device=None, batch_size=64, normalize=False):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = None

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        return self.model.encode(
            list(X),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )




base_claim = Pipeline(steps=[
    ("embed", SentenceEncoder(model_name=embed_model_name, normalize=False)),
    ("scale", StandardScaler(with_mean=False)),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=120,
        early_stopping=True,
        random_state=42
    ))
])

claim_pipe = CalibratedClassifierCV(
    base_estimator=base_claim,
    method="isotonic",
    cv=3
)

print("[TRAIN] Fitting CLAIM (ST+MLP+Calibration) ...")
claim_pipe.fit(X_train, y_train)

print("[EVAL] CLAIM calibrated probabilities on X_test ...")
y_prob = claim_pipe.predict_proba(X_test)[:, 1]


thresholds = np.linspace(0.05, 0.95, 19)
min_precision = 0.60  
best_thr = 0.5
best_recall = -1.0
best_prec = 0.0
best_f1 = 0.0

for thr in thresholds:
    y_tmp = (y_prob >= thr).astype(int)
    rec = recall_score(y_test, y_tmp, zero_division=0)
    prec = precision_score(y_test, y_tmp, zero_division=0)
    f1 = f1_score(y_test, y_tmp, zero_division=0)
    if prec >= min_precision and rec > best_recall:
        best_recall, best_prec, best_thr, best_f1 = rec, prec, thr, f1

if best_recall < 0:
    best_thr = 0.5
    print("[THRESH] No threshold met the precision floor. Falling back to 0.50")
else:
    print(f"[THRESH] CLAIM best by Recall@Prec≥{min_precision:.2f}: "
          f"thr={best_thr:.2f} (Recall={best_recall:.3f}, Precision={best_prec:.3f}, F1={best_f1:.3f})")

y_pred = (y_prob >= best_thr).astype(int)
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print(f"[METRICS] CLAIM Accuracy@thr: {acc:.3f}")
print(f"[METRICS] CLAIM ROC-AUC: {roc:.3f}")
print("\n[REPORT]\n", classification_report(y_test, y_pred, target_names=["opinion(0)", "claim(1)"]))
print("[CONFUSION]\n", confusion_matrix(y_test, y_pred))



sent_pipe = None
labels_sentiment = None
sent_metrics = None

try:
    df_sent = pd.read_sql(
        """
        SELECT video_transcription_text, sentiment_label
        FROM tiktok_csv
        WHERE video_transcription_text IS NOT NULL
          AND sentiment_label IS NOT NULL
        """,
        engine
    ).dropna()

    if not df_sent.empty:
        yS = df_sent["sentiment_label"].astype(str)   # לדוגמה: positive/neutral/negative
        XS = df_sent["video_transcription_text"].astype(str)

        XS_train, XS_test, yS_train, yS_test = train_test_split(
            XS, yS, test_size=0.2, stratify=yS, random_state=42
        )

        base_sent = Pipeline(steps=[
            ("embed", SentenceEncoder(model_name=embed_model_name, normalize=False)),
            ("scale", StandardScaler(with_mean=False)),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(192, 96),
                activation="relu",
                solver="adam",
                learning_rate_init=1e-3,
                alpha=1e-4,
                max_iter=120,
                early_stopping=True,
                random_state=42
            ))
        ])

        sent_pipe_cal = CalibratedClassifierCV(
            base_estimator=base_sent,
            method="isotonic",
            cv=3
        )

        print("[TRAIN] Fitting SENTIMENT (ST+MLP+Calibration) ...")
        sent_pipe_cal.fit(XS_train, yS_train)

        yS_prob = sent_pipe_cal.predict_proba(XS_test)
        yS_pred = sent_pipe_cal.predict(XS_test)

        
        sent_acc = accuracy_score(yS_test, yS_pred)
        sent_f1_macro = f1_score(yS_test, yS_pred, average="macro", zero_division=0)
        print(f"[METRICS] SENTIMENT Acc: {sent_acc:.3f} | Macro-F1: {sent_f1_macro:.3f}")

       
        labels_sentiment = list(getattr(sent_pipe_cal, "classes_", []))
        sent_pipe = sent_pipe_cal
        sent_metrics = {"accuracy": float(sent_acc), "macro_f1": float(sent_f1_macro)}

    else:
        print("[SENTIMENT] No labeled rows. Skipping sentiment training.")

except Exception as e:
    print("[SENTIMENT] Skipped (reason):", e)



bundle = {
    "claim_pipe": claim_pipe,
    "embed_model_name": embed_model_name,
    "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    "calibration": {"method": "isotonic", "cv": 3},
    "decision_threshold": float(best_thr),
    "decision_target": {"optimize": "recall", "precision_floor": float(min_precision)},
    "versions": {
        "sklearn": sklearn.__version__,
        "sentence_transformers": st_pkg.__version__
    },
    "metrics": {
        "claim": {
            "accuracy_at_thr": float(acc),
            "roc_auc": float(roc),
            "recall_at_thr": float(recall_score(y_test, y_pred, zero_division=0)),
            "precision_at_thr": float(precision_score(y_test, y_pred, zero_division=0)),
            "f1_at_thr": float(f1_score(y_test, y_pred, zero_division=0))
        }
    }
}

# הוספת סנטימנט לבאנדל רק אם אומן בהצלחה
if sent_pipe is not None and labels_sentiment:
    bundle["sent_pipe"] = sent_pipe
    bundle["labels_sentiment"] = labels_sentiment
    bundle["metrics"]["sentiment"] = sent_metrics

MODEL_PATH = "model_pipeline.pkl"
joblib.dump(bundle, MODEL_PATH)
print(f"[SAVE] Bundle saved to {MODEL_PATH}")

PROD_MODEL_PATH = "model_pipeline_prod.pkl"
joblib.dump(bundle, PROD_MODEL_PATH)
print(f"[SAVE] Bundle saved to {PROD_MODEL_PATH}")


#with engine.begin() as conn:
    #conn.execute(sql_text("""
        #CREATE TABLE IF NOT EXISTS model_store (
            #id BIGSERIAL PRIMARY KEY,
            #model_bytes BYTEA NOT NULL,
            #created_at TIMESTAMPTZ DEFAULT NOW()
        #);
    #"""))

#model_blob = pickle.dumps(bundle)
#with engine.begin() as conn:
    #conn.execute(
       # sql_text("INSERT INTO model_store (model_bytes) VALUES (:b)"),
        #{"b": model_blob}
   ## )

#print("[DB] Model stored in table: model_store")
#print("[DONE] Training & storage complete.")
