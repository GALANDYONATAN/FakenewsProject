function transcribe() {
  const url = document.getElementById('videoUrl').value;
  const resultDiv = document.getElementById('result'); 
  const feature = document.getElementById('featureSelect').value; 
  resultDiv.style.display = 'block';
  resultDiv.innerText = "⏳ Please wait, this might take a few moments...";
  fetch('/transcribe', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: url, feature: feature })
  })
  .then(res => res.json())
  .then(data => {
        document.getElementById('result').innerText = `Fake news check: ${data["Fake news check"]}\n` +
        `Reliability: ${data["Reliability"]}%\n` +
        `Unreliability: ${data["Unreliability"]}%`;
})
  .catch(err => {
        document.getElementById('result').innerText = "❌ Error during process: " + err;
});
}


