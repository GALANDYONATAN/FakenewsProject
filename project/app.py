from flask import Flask, render_template, request, jsonify
import yt_dlp, ffmpeg, whisper, ssl, os
import pandas as pd
import pickle
from sqlalchemy import create_engine, text as sql_text
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

import os




ssl._create_default_https_context = ssl._create_unverified_context
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline_prod.pkl")
MODEL_DIR  = os.path.join(BASE_DIR, "models")

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
            normalize_embeddings=self.normalize,
            show_progress_bar=False
        )

DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})

#def load_model_from_db_or_file():
    #try:
        #with engine.begin() as conn:
            #row = conn.execute(
                #sql_text("SELECT model_bytes FROM model_store ORDER BY id DESC LIMIT 1")
            #).first()
        #if row and row[0]:
           # print("[BOOT] Loaded model from DB")
            #return pickle.loads(row[0])
    #except Exception as e:
        #print("[WARN] DB model load failed:", e)
    #print("[INFO] Falling back to PKL file:", MODEL_PATH)
    #return joblib.load(MODEL_PATH)

pipe = joblib.load(MODEL_PATH)

def _resolve_clf_and_threshold(pipe_obj):
   
    if isinstance(pipe_obj, dict):
        clf_claim   = pipe_obj.get("claim_pipe", pipe_obj)
        thr         = float(pipe_obj.get("decision_threshold", 0.5))
        sent        = pipe_obj.get("sent_pipe")
        sent_labels = pipe_obj.get("labels_sentiment")
        return clf_claim, thr, sent, sent_labels
    return pipe_obj, 0.5, None, None

def check_text(text: str):
    clf_claim, thr, sent_clf, sent_labels = _resolve_clf_and_threshold(pipe)
   
    proba_vec = clf_claim.predict_proba([text])[0]
    try:
        classes = list(getattr(clf_claim, "classes_", [0, 1]))
        idx_pos = classes.index(1) if 1 in classes else 1
    except Exception:
        idx_pos = 1
    p_claim = float(proba_vec[idx_pos])
    is_claim = (p_claim >= thr)
    result = {
        "fake_news_check": bool(is_claim),
        "reliability": round(p_claim * 100, 2),            
        "unreliability": round((1.0 - p_claim) * 100, 2)   
    }
    
    if sent_clf is not None:
        s_proba = sent_clf.predict_proba([text])[0]
        s_classes = list(getattr(sent_clf, "classes_", []))
        if s_classes:
            s_idx = int(np.argmax(s_proba))
            s_label = s_classes[s_idx]
            s_conf  = float(s_proba[s_idx])
            result["sentiment_label"] = str(s_label)
            result["sentiment_confidence"] = round(s_conf * 100, 2)
        else:
            s_idx = int(np.argmax(s_proba))
            result["sentiment_label"] = "unknown"
            result["sentiment_confidence"] = round(float(s_proba[s_idx]) * 100, 2)
    return result


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/videopage")
def videopage():
    return render_template("videopage.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/docs")
def docs():
    return render_template("docs.html")

@app.route("/presentation")
def presentation():
    return render_template("presentation.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/howto")
def howto():
    return render_template("howto.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/db_test")
def db_test():
    try:
        with engine.begin() as conn:
            count = conn.execute(sql_text("SELECT COUNT(*) FROM tiktok_data")).scalar()
        return jsonify({"rows_in_tiktok_data": int(count)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        video_url = data.get("url")

     
        for f in ['video.mp4', 'audio.wav']:
            if os.path.exists(f):
                os.remove(f)

      
        ydl_opts = {'outtmpl': 'video.mp4', 'format': 'mp4'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

  
        ffmpeg.input('video.mp4').output(
            'audio.wav', format='wav',
            acodec='pcm_s16le', ac=1, ar='16000'
        ).overwrite_output().run()

     
        model = whisper.load_model("base", download_root=MODEL_DIR)
        result = model.transcribe("audio.wav", language="en")
        transcription = (result.get("text") or "").strip()
        if not transcription:
            return jsonify({"error": "No text received from transcription"}), 400

      
        pred = check_text(transcription)
       

      
        upsert_sql = sql_text("""
            INSERT INTO tiktok_data
                (video_url, transcription, fake_news_check, reliability, unreliability, source)
            VALUES
                (:u, :t, :f, :r, :ur, :s)
            ON CONFLICT (video_url) DO UPDATE
            SET
                video_url       = EXCLUDED.video_url,
                fake_news_check = EXCLUDED.fake_news_check,
                reliability     = EXCLUDED.reliability,
                unreliability   = EXCLUDED.unreliability,
                source          = EXCLUDED.source,
                updated_at      = NOW()
            RETURNING
                id, video_url, transcription, fake_news_check, reliability, unreliability, source, created_at, updated_at
        """)
        with engine.begin() as conn:
            saved = conn.execute(upsert_sql, {
                "u": video_url,
                "t": transcription,
                "f": pred["fake_news_check"],
                "r": pred["reliability"],
                "ur": pred["unreliability"],
                "s": "model"
            }).mappings().first()

        
        return jsonify({
            "Fake news check": "True" if saved["fake_news_check"] else "False",
            "Reliability": float(saved["reliability"]),
            "Unreliability": float(saved["unreliability"]),
            "transcription": transcription,
           
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

