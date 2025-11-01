import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split


# חיבור למסד הנתונים
DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})


df = pd.read_sql("SELECT video_transcription_text, claim_status FROM tiktok_csv", engine)
df = df.dropna(subset=["video_transcription_text", "claim_status"])

# המרת תגיות למספרים
y = df["claim_status"].replace({"claim": 1, "opinion": 0})
X = df["video_transcription_text"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)




count_vec = CountVectorizer(ngram_range=(1, 3),
                            max_features=5000,
                            stop_words='english')

# בניית pipeline לאימון
pipe = Pipeline([
    ("cv", count_vec),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

cv_params = {"rf__n_estimators": [75, 100, 200],
            "rf__max_depth": [5, 7, None],
             "rf__max_features": [0.3, 0.6],
             "rf__max_samples": [0.7],          # אפשר גם להסיר אם אין צורך בסאב־סמפול
             "rf__min_samples_leaf": [1, 2],
             "rf__min_samples_split": [2, 3],
             
             }

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


rf_cv = GridSearchCV(
    estimator=pipe,
    param_grid=cv_params,
    scoring=scoring,
    refit="recall",   
    cv=5,
    verbose=2

)

# אימון המודל
rf_cv.fit(X_train, y_train)
# שמירה לקובץ  
best_pipe = rf_cv.best_estimator_

MODEL_PATH = "model_pipeline.pkl"
joblib.dump(best_pipe, MODEL_PATH)
print(f" Model saved to {MODEL_PATH}")

print(" Model trained successfully")
