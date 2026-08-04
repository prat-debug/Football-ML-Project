// ==========================================================
// GET HTML ELEMENTS
// ==========================================================

const clusteringForm = document.getElementById("clusteringForm");

const clusterResult = document.getElementById("clusterResult");

const clusterError = document.getElementById("clusterError");

const category = document.getElementById("category");

const clusterId = document.getElementById("clusterId");

// ==========================================================
// FORM SUBMIT
// ==========================================================

clusteringForm.addEventListener(
  "submit",

  async function (event) {
    event.preventDefault();

    // --------------------------------------------------
    // Reset previous output
    // --------------------------------------------------

    clusterResult.style.display = "none";

    clusterError.style.display = "none";

    // --------------------------------------------------
    // Read Input Values
    // --------------------------------------------------

    const formData = new FormData();

    formData.append("age", document.getElementById("age").value);

    formData.append("height", document.getElementById("height").value);

    formData.append("matches", document.getElementById("matches").value);

    formData.append("minutes", document.getElementById("minutes").value);

    formData.append(
      "goals_per_90",
      document.getElementById("goals_per_90").value,
    );

    formData.append(
      "assists_per_90",
      document.getElementById("assists_per_90").value,
    );

    formData.append(
      "estimated_xg",
      document.getElementById("estimated_xg").value,
    );

    formData.append(
      "estimated_xa",
      document.getElementById("estimated_xa").value,
    );

    formData.append(
      "physicality_score",
      document.getElementById("physicality_score").value,
    );

    // --------------------------------------------------
    // Send Request
    // --------------------------------------------------

    try {
      const response = await fetch(
        "/predict-cluster",

        {
          method: "POST",

          body: formData,
        },
      );

      const data = await response.json();

      // ----------------------------------------------
      // Backend Error
      // ----------------------------------------------

      if (data.error) {
        clusterError.innerHTML = data.error;

        clusterError.style.display = "block";

        return;
      }

      // ----------------------------------------------
      // Display Prediction
      // ----------------------------------------------

      category.innerHTML = "⭐ " + data.category;

      clusterId.innerHTML = data.cluster_id;

      clusterResult.style.display = "block";
    } catch (error) {
      clusterError.innerHTML =
        "Something went wrong while predicting the cluster.";

      clusterError.style.display = "block";
    }
  },
);
