const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const placeholderView = document.getElementById("placeholderView");
const previewView = document.getElementById("previewView");
const previewImage = document.getElementById("previewImage");
const imageMeta = document.getElementById("imageMeta");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeBtnText = document.getElementById("analyzeBtnText");
const loadingSpinner = document.getElementById("loadingSpinner");
const resetBtn = document.getElementById("resetBtn");
const errorBox = document.getElementById("errorBox");

const modelBadgeText = document.getElementById("modelBadgeText");
const thresholdChips = document.getElementById("thresholdChips");
const latencyBadge = document.getElementById("latencyBadge");
const emptyResults = document.getElementById("emptyResults");
const resultsContainer = document.getElementById("resultsContainer");
const summaryBanner = document.getElementById("summaryBanner");
const predictionsList = document.getElementById("predictionsList");

let selectedFile = null;

// Load model & threshold configuration from backend on startup
async function initConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error("Failed to load model config");
    const cfg = await res.json();

    modelBadgeText.textContent = `${cfg.backbone} · ${cfg.img_size}×${cfg.img_size} · ${cfg.model_file}`;

    thresholdChips.innerHTML = cfg.classes
      .map(
        (c) => `
        <div class="chip">
          <span>${c.label}</span>
          <span>≥ ${(c.threshold * 100).toFixed(0)}%</span>
        </div>`
      )
      .join("");
  } catch (err) {
    modelBadgeText.textContent = "Config unavailable";
    showError(err.message);
  }
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function handleFileSelect(file) {
  clearError();
  if (!file || !file.type.startsWith("image/")) {
    showError("Please select a valid image file (JPG, PNG, or WEBP).");
    return;
  }

  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
    previewImage.onload = () => {
      const sizeKB = (file.size / 1024).toFixed(1);
      imageMeta.innerHTML = `
        <span>${file.name}</span>
        <span>${previewImage.naturalWidth}×${previewImage.naturalHeight}px · ${sizeKB} KB</span>
      `;
    };
    placeholderView.hidden = true;
    previewView.hidden = false;
    resetBtn.hidden = false;
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function resetUI() {
  selectedFile = null;
  fileInput.value = "";
  previewImage.src = "";
  placeholderView.hidden = false;
  previewView.hidden = true;
  resetBtn.hidden = true;
  analyzeBtn.disabled = true;
  latencyBadge.hidden = true;
  resultsContainer.hidden = true;
  emptyResults.hidden = false;
  clearError();
}

// Dropzone & input listeners
dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileSelect(e.target.files[0]);
  }
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-over");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("drag-over");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  if (e.dataTransfer.files.length > 0) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

resetBtn.addEventListener("click", resetUI);

// Submit image to /api/predict
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  clearError();
  analyzeBtn.disabled = true;
  analyzeBtnText.textContent = "Analyzing Imagery...";
  loadingSpinner.hidden = false;

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtnText.textContent = "Run Anomaly Detection";
    loadingSpinner.hidden = true;
  }
});

function renderResults(data) {
  emptyResults.hidden = true;
  resultsContainer.hidden = false;

  latencyBadge.textContent = `Inference: ${data.inference_ms} ms`;
  latencyBadge.hidden = false;

  if (data.detected_count > 0) {
    summaryBanner.className = "summary-banner anomaly";
    summaryBanner.textContent = `${data.detected_count} Field Anomal${
      data.detected_count === 1 ? "y" : "ies"
    } Detected Above Calibrated Threshold`;
  } else {
    summaryBanner.className = "summary-banner healthy";
    summaryBanner.textContent =
      "No Anomalies Detected — All class scores are below their calibrated thresholds.";
  }

  predictionsList.innerHTML = data.predictions
    .map((item) => {
      const stateClass = item.detected ? "detected" : "clear";
      const badgeLabel = item.detected ? "Detected" : "Below Threshold";

      return `
        <div class="pred-item ${item.detected ? "is-detected" : ""}">
          <div class="pred-top">
            <div class="pred-title">
              <span>${item.label}</span>
              <span class="status-tag ${stateClass}">${badgeLabel}</span>
            </div>
            <div class="pred-scores">
              <strong>${item.confidence_pct}%</strong>
              <small>(Threshold: ${item.threshold_pct}%)</small>
            </div>
          </div>

          <div class="bar-track" title="White tick marks the ${item.threshold_pct}% threshold">
            <div class="bar-fill ${stateClass}" style="width: ${Math.min(
              item.confidence_pct,
              100
            )}%"></div>
            <div class="threshold-marker" style="left: ${item.threshold_pct}%"></div>
          </div>

          <p class="pred-desc">${item.description}</p>
          ${
            item.detected
              ? `<p class="pred-action"><strong>Recommended Action:</strong> ${item.action}</p>`
              : ""
          }
        </div>
      `;
    })
    .join("");
}

initConfig();
