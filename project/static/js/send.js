function send() {
  const Text = document.getElementById('Text').value;
  const Textresult = document.getElementById('Textresult'); 
  Textresult.style.display = 'block';
  Textresult.innerText = "⏳ Please wait, this might take a few moments...";
  fetch('/send', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({ Text: Text })
  })
  .then(res => res.json())
  .then(data => {
        document.getElementById('Textresult').innerText =`Fake news check: ${data["Fake news check"]}\n` +
        `Reliability: ${data["Reliability"]}%\n` +
        `Unreliability: ${data["Unreliability"]}%`;
})
  .catch(err => {
        document.getElementById('Textresult').innerText = "❌ Error during process: " + err;
});
}
