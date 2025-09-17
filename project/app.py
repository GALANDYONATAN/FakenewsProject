from flask import Flask, render_template, request, jsonify
import yt_dlp, ffmpeg, whisper, ssl, os
import webbrowser
from threading import Timer

# טיפול בתעודת SSL
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")


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
    webbrowser.open_new("http://127.0.0.1:5001/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True,port=5001)


