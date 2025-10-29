from flask import Flask, render_template, request, jsonify
import yt_dlp, ffmpeg, whisper, ssl, os
import pandas as pd

# טיפול בתעודת SSL
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)




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


# ==== CSV load ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR,"tiktok_predictions_full.csv")
df = pd.read_csv(CSV_PATH)



@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        video_url = data.get("url")
        feature = data.get("feature")




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
        textResult = result["text"].strip()


        



        if not textResult:
            return jsonify({"error": "No text received"}), 400

        # חיפוש טקסט במסד
        row = df[df["text"].str.strip()== textResult]
        if row.empty:
            return jsonify({"error": "Text not found in dataset"}), 404

        # לוקחים את הערכים מהשורה הראשונה שנמצאה
        row = row.iloc[0]
        return jsonify({
            "Fake news check": str(row["Fake news check"]),
            "Reliability": float(row["Reliability"]),
            "Unreliability": float(row["Unreliability"])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== Local run (Render uses gunicorn instead) ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
