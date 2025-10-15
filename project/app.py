from flask import Flask, render_template, request, jsonify
import yt_dlp, ffmpeg, whisper, ssl, os
import pandas as pd

# טיפול בתעודת SSL
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

# ==== Whisper model loaded once ====
whisper_model = whisper.load_model("base")

# ==== CSV load ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "tik_tok_predictions_full_backup.csv")
df = pd.read_csv(CSV_PATH)

# ==== Routes ====
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/docs")
def docs():
    return render_template("docs.html")

@app.route("/presentation")
def presentation():
    return render_template("presentation.html")

@app.route('/send', methods=['POST'])
def send():
    data = request.get_json(force=True)
    text = (data.get("Text") or "").strip()
    if not text:
        return jsonify({"error": "No text received"}), 400
    
    row = df[df["text_snippet"].str.strip() == text]
    if row.empty:
        return jsonify({"error": "Text not found in dataset"}), 404
    
    row = row.iloc[0]
    return jsonify({
        "Fake news check": str(row["Fake news check"]),
        "Reliability": float(row["Reliability"]),
        "Unreliability": float(row["Unreliability"])
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        video_url = data.get("url")

        for f in ['video.mp4', 'audio.wav']:
            if os.path.exists(f):
                os.remove(f)

        # Download video
        ydl_opts = {'outtmpl': 'video.mp4', 'format': 'mp4'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Extract audio
        ffmpeg.input('video.mp4').output(
            'audio.wav', format='wav',
            acodec='pcm_s16le', ac=1, ar='16000'
        ).overwrite_output().run()

        # Transcribe
        result = whisper_model.transcribe("audio.wav", language="en")
        return jsonify({"transcription": result["text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== Local run (Render uses gunicorn instead) ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
