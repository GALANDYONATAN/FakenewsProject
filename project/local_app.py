from flask import Flask, render_template, request, jsonify
import yt_dlp, ffmpeg, whisper, ssl, os
import webbrowser
from threading import Timer
import pandas as pd


# טיפול בתעודת SSL
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)



@app.route("/")
def home():
    return render_template("home.html")


@app.route("/textpage")
def textpage():
    return render_template("textpage.html")


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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # התיקייה של app.py (כלומר project/)
CSV_PATH = os.path.join(BASE_DIR, "tik_tok_predictions_full_backup.csv")
df = pd.read_csv(CSV_PATH)

@app.route('/send', methods=['POST'])
def send():
   
    data=request.get_json(force=True)
    text = (data.get("Text") or "").strip()
    if not text:
         return jsonify({"error": "No text received"}), 400
    row = df[df["text_snippet"].str.strip() == text]
    if row.empty:
        return jsonify({"error": "Text not found in dataset"}), 404
        # לוקחים את הערכים מהשורה הראשונה שנמצאה
    row = row.iloc[0]
    return jsonify({
        "Fake news check": str(row["Fake news check"]),
        "Reliability": float(row["Reliability"]),
        "Unreliability": float(row["Unreliability"])
    })
#if __name__ == "__main__":
    #app.run(debug=True)



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
        return jsonify({"transcription": result["text"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# תוספת לפתיחת הדפדפן אוטומטית
def open_browser():
    webbrowser.open_new("http://0.0.0.0:5002/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True,port=5002)
#if __name__ == '__main__':
    #app.run(host="0.0.0.0", port=5000)
