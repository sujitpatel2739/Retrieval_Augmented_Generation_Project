async function submitQuery() {
  const query = document.getElementById("query").value;
  const top_k = parseInt(document.getElementById("top_k").value);
  const resultDiv = document.getElementById("queryResult");
  resultDiv.innerHTML = "Processing...";

  const response = await fetch("/query", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ query, top_k })
  });

  const data = await response.json();
  if (response.ok && data.answer) {
    resultDiv.innerHTML = `✅ <b>Answer:</b> ${data.answer}<br><b>Confidence:</b> ${data.confidence_score}`;
  } else {
    resultDiv.innerHTML = `❌ <b>Error:</b> ${data.detail || 'Unknown error.'}`;
  }
}

async function uploadDocument() {
  const fileInput = document.getElementById("docFile");
  const min_token_len = parseInt(document.getElementById("min_token_len").value);
  const max_token_len = parseInt(document.getElementById("max_token_len").value);
  const resultDiv = document.getElementById("uploadResult");

  if (!fileInput.files.length) {
    resultDiv.innerHTML = "❌ Please select a file first.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("min_token_len", min_token_len);
  formData.append("max_token_len", max_token_len);

  resultDiv.innerHTML = "Uploading...";

  const response = await fetch("/document", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  if (response.ok && data.status !== "ERROR") {
    resultDiv.innerHTML = `✅ Document processed successfully.`;
  } else {
    resultDiv.innerHTML = `❌ <b>Error:</b> ${data.detail || 'Unknown error.'}`;
  }
}
