// ==========================================================
// GET HTML ELEMENTS
// ==========================================================

const form = document.getElementById("recommendationForm");

const resultsSection = document.getElementById("resultsSection");

const tableHeader = document.getElementById("tableHeader");

const tableBody = document.getElementById("tableBody");

const errorMessage = document.getElementById("errorMessage");

// ==========================================================
// FORM SUBMIT
// ==========================================================

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  // Hide previous results

  resultsSection.style.display = "none";

  errorMessage.style.display = "none";

  tableHeader.innerHTML = "";

  tableBody.innerHTML = "";

  // Read Inputs

  const player_name = document.getElementById("player_name").value;

  const method = document.getElementById("method").value;

  const top_n = document.getElementById("top_n").value;

  // Create Form Data

  const formData = new FormData();

  formData.append("player_name", player_name);

  formData.append("method", method);

  formData.append("top_n", top_n);

  try {
    const response = await fetch("/recommend", {
      method: "POST",

      body: formData,
    });

    const data = await response.json();

    // Check for Backend Error

    if (data.error) {
      errorMessage.innerHTML = data.error;

      errorMessage.style.display = "block";

      return;
    }

    // Empty Result

    if (data.length === 0) {
      errorMessage.innerHTML = "No recommendations found.";

      errorMessage.style.display = "block";

      return;
    }

    // ==========================================================
    // TABLE HEADER
    // ==========================================================

    let metricColumn = "Distance";

    if (method === "cosine") {
      metricColumn = "Similarity";
    }

    tableHeader.innerHTML = `

            <th>Rank</th>

            <th>Player</th>

            <th>${metricColumn}</th>

        `;

    // ==========================================================
    // TABLE BODY
    // ==========================================================

    data.forEach(function (row) {
      let metricValue = "";

      if (method === "knn") {
        metricValue = row.Distance;
      } else {
        metricValue = row.Similarity;
      }

      tableBody.innerHTML += `

                <tr>

                    <td>${row.Rank}</td>

                    <td>${row.Player}</td>

                    <td>${metricValue}</td>

                </tr>

            `;
    });

    resultsSection.style.display = "block";
  } catch (error) {
    errorMessage.innerHTML =
      "Something went wrong while fetching recommendations.";

    errorMessage.style.display = "block";
  }
});
