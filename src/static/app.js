async function sendQuery() {
  const query = document.getElementById("queryInput").value;
  const responseArea = document.getElementById("responseArea");

  responseArea.innerHTML = "Processing...";

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query })
    });

    if (!response.ok) {
      throw new Error("Failed to get response");
    }

    const data = await response.json();
    responseArea.innerHTML = `<strong>Answer:</strong> ${data.answer}`;
  } catch (error) {
    responseArea.innerHTML = `Error: ${error.message}`;
  }
}
