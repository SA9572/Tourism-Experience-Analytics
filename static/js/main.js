document.addEventListener("DOMContentLoaded", () => {
  if (typeof Chart === "undefined" || !window.edaData) return;

  const { continent_labels, continent_counts, type_labels, type_counts, top_attraction_labels, top_attraction_ratings } =
    window.edaData;

  const continentCtx = document.getElementById("continentChart");
  if (continentCtx && continent_labels && continent_labels.length) {
    new Chart(continentCtx, {
      type: "bar",
      data: {
        labels: continent_labels,
        datasets: [
          {
            label: "Users",
            data: continent_counts,
            backgroundColor: "rgba(59, 130, 246, 0.6)",
            borderColor: "rgb(37, 99, 235)",
            borderWidth: 1,
          },
        ],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: "#cbd5f5" },
            grid: { display: false },
          },
          y: {
            ticks: { color: "#cbd5f5" },
            grid: { color: "rgba(51, 65, 85, 0.7)" },
          },
        },
      },
    });
  }

  const typeCtx = document.getElementById("typeChart");
  if (typeCtx && type_labels && type_labels.length) {
    new Chart(typeCtx, {
      type: "doughnut",
      data: {
        labels: type_labels,
        datasets: [
          {
            data: type_counts,
            backgroundColor: [
              "rgba(56, 189, 248, 0.8)",
              "rgba(129, 140, 248, 0.8)",
              "rgba(244, 114, 182, 0.8)",
              "rgba(252, 211, 77, 0.8)",
              "rgba(52, 211, 153, 0.8)",
            ],
            borderColor: "#020617",
            borderWidth: 2,
          },
        ],
      },
      options: {
        plugins: {
          legend: {
            labels: { color: "#cbd5f5" },
          },
        },
      },
    });
  }

  const topAttrCtx = document.getElementById("topAttractionChart");
  if (topAttrCtx && top_attraction_labels && top_attraction_labels.length) {
    new Chart(topAttrCtx, {
      type: "bar",
      data: {
        labels: top_attraction_labels,
        datasets: [
          {
            label: "Average Rating",
            data: top_attraction_ratings,
            backgroundColor: "rgba(248, 250, 252, 0.9)",
            borderColor: "#facc15",
            borderWidth: 1,
          },
        ],
      },
      options: {
        indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: "#cbd5f5" },
            grid: { color: "rgba(51, 65, 85, 0.7)" },
          },
          y: {
            ticks: { color: "#cbd5f5" },
            grid: { display: false },
          },
        },
      },
    });
  }
});
