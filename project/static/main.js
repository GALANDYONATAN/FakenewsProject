function transcribe() {
  const url = document.getElementById('videoUrl').value;

  fetch('/transcribe', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: url })
  })
  .then(res => res.json())
  .then(data => {
      document.getElementById('result').innerText = data.transcription || "לא התקבל תמלול";
  })
  .catch(err => {
      document.getElementById('result').innerText = "❌ שגיאה בתהליך: " + err;
  });
}
