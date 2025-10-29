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
      if (data.error) {
          resultDiv.innerText = "❌ Error: " + data.error;
          return;
      }

      resultDiv.innerText = 
          `Transcription: ${data.transcription || "N/A"}\n\n` +
          ` Fake news check: ${data["Fake news check"]}\n` +
          ` Reliability: ${data["Reliability"]}%\n` +
          ` Unreliability: ${data["Unreliability"]}%\n\n`;

        })
          
   .catch(err => {
      resultDiv.innerText = "❌ Error during process: " + err;
  });
}


