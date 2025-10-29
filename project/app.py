from flask import Flask, render_template, request, jsonify
import yt_dlp, ffmpeg, whisper, ssl, os
import pandas as pd
from sqlalchemy import create_engine, text as sql_text
import joblib

# טיפול בתעודת SSL
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.pkl")   # <<< ודא שקובץ זה קיים בצדך
pipe = joblib.load(MODEL_PATH)


DATABASE_URL = os.environ["DATABASE_URL"]
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})


def check_text(text: str):
    proba = pipe.predict_proba([text])[0]  # [P(opinion=0), P(claim=1)]
    is_claim = (proba[1] >= 0.5)
    return {
        "fake_news_check": bool(is_claim),
        "reliability": round(float(proba[1]) * 100, 2),
        "unreliability": round(float(proba[0]) * 100, 2)
    }


# ==== Routes ====
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

        # הורדת וידאו
        ydl_opts = {'outtmpl': 'video.mp4', 'format': 'mp4'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # חילוץ אודיו
        ffmpeg.input('video.mp4').output(
            'audio.wav', format='wav',
            acodec='pcm_s16le', ac=1, ar='16000'
        ).overwrite_output().run()

        # תמלול
        model = whisper.load_model("base")
        result = model.transcribe("audio.wav", language="en")
        transcription = (result.get("text") or "").strip()
        if not transcription:
            return jsonify({"error": "No text received from transcription"}), 400
        
        pred = check_text(transcription)



        # 5) UPSERT למסד והחזרה ממנו
        upsert_sql = sql_text("""
            INSERT INTO tiktok_data
                (video_url, transcription, fake_news_check, reliability, unreliability, source)
            VALUES
                (:u, :t, :f, :r, :ur, :s)
            ON CONFLICT (transcription) DO UPDATE
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
            "source": saved["source"],
            "id": int(saved["id"]),
            "video_url": saved["video_url"],
            "transcription": saved["transcription"],
            "Fake news check": "True" if saved["fake_news_check"] else "False",
            "Reliability": float(saved["reliability"]),
            "Unreliability": float(saved["unreliability"]),
            "created_at": str(saved["created_at"]),
            "updated_at": str(saved["updated_at"])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    




# ==== Local run (Render uses gunicorn instead) ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
