import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import clone


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


y_pred = rf_cv.predict(X_test)
y_proba = rf_cv.predict_proba(X_test)[:, 1]  # להערכת ROC-AUC

print("\nBest params (CV):", rf_cv.best_params_)
print("Best CV recall:", rf_cv.best_score_)

print("\n=== Test classification report ===")
print(classification_report(y_test, y_pred, digits=3))

print("\n=== Test confusion matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\nTest ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))


# שמירה לקובץ  
best_pipe = rf_cv.best_estimator_


MODEL_PATH = "model_pipeline.pkl"
joblib.dump(best_pipe, MODEL_PATH)
print(f" Model saved to {MODEL_PATH}")

print(" Model trained successfully")


final_pipe = clone(best_pipe)
final_pipe.fit(X, y)

PROD_MODEL_PATH = "model_pipeline_prod.pkl"
joblib.dump(final_pipe, PROD_MODEL_PATH)
print(f" Production model saved to {PROD_MODEL_PATH}")
