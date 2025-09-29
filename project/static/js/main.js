function transcribe() {
  const url = document.getElementById('videoUrl').value;
  const resultDiv = document.getElementById('result'); 
  resultDiv.style.display = 'block';
  resultDiv.innerText = "⏳ Please wait, this might take a few moments...";
  fetch('/transcribe', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: url })
  })
  .then(res => res.json())
  .then(data => {
        document.getElementById('result').innerText = data.transcription || "No transcription received.";
})
  .catch(err => {
        document.getElementById('result').innerText = "❌ Error during process: " + err;
});
}
