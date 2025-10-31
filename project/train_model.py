import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# חיבור למסד הנתונים
DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})

# שליפת נתונים מהטבלה המעודכנת
df = pd.read_sql("SELECT video_transcription_text, claim_status FROM tiktok_csv", engine)
df = df.dropna(subset=["video_transcription_text", "claim_status"])

# המרת תגיות למספרים
y = df["claim_status"].replace({"claim": 1, "opinion": 0})
X = df["video_transcription_text"].astype(str)

# בניית pipeline לאימון
pipe = Pipeline([
    ("cv", CountVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# אימון המודל
pipe.fit(X, y)
print(" Model trained successfully")

# שמירה לקובץ
MODEL_PATH = "model_pipeline.pkl"
joblib.dump(pipe, MODEL_PATH)
print(f" Model saved to {MODEL_PATH}")
