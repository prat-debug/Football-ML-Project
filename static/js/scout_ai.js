document
  .getElementById("scoutAIForm")
  .addEventListener("submit", async function (event) {
    event.preventDefault();

    const question = document.getElementById("question").value.trim();

    const errorBox = document.getElementById("chatError");

    const resultCard = document.getElementById("chatResult");

    const answerBox = document.getElementById("chatAnswer");

    errorBox.style.display = "none";

    resultCard.style.display = "none";

    try {
      const response = await fetch("/ask-scout-ai", {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          question: question,
        }),
      });

      const data = await response.json();

      if (!data.success) {
        errorBox.innerText = data.error;

        errorBox.style.display = "block";

        return;
      }

      answerBox.innerText = data.answer;

      resultCard.style.display = "block";
    } catch (error) {
      errorBox.innerText = "Unable to contact Scout AI.";

      errorBox.style.display = "block";
    }
  });
