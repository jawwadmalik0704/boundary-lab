/* =========================================================
   BOUNDARYLAB
   APPLICATION CONTROLLER
========================================================= */

/* =========================================================
   DOM ELEMENTS
========================================================= */

const trainButton = document.getElementById("trainButton");

const resetModelButton = document.getElementById("resetModelButton");

const class0Button = document.getElementById("class0Button");

const class1Button = document.getElementById("class1Button");

const clearDataButton = document.getElementById("clearDataButton");

const learningRateInput = document.getElementById("learningRate");

const epochsInput = document.getElementById("epochs");

const trainingSpeedSelect = document.getElementById("trainingSpeed");

const selectedClassDisplay = document.getElementById("selectedClassDisplay");

const epochDisplay = document.getElementById("epochDisplay");

const liveLoss = document.getElementById("liveLoss");

const trainingState = document.getElementById("trainingState");

/* =========================================================
   STATE
========================================================= */

let selectedClass = 0;

let currentModel = createInitialModel();

let isTraining = false;

let animationId = null;

/*
 * The original six-point dataset is preserved.
 */

const originalTrainingData = trainingData.map((sample) => ({
  x: [...sample.x],

  y: sample.y,
}));

/* =========================================================
   SPEED
========================================================= */

function getEpochsPerFrame() {
  const speed = trainingSpeedSelect.value;

  if (speed === "slow") {
    return 1;
  }

  if (speed === "fast") {
    return 8;
  }

  return 3;
}

/* =========================================================
   CLASS SELECTION
========================================================= */

function selectClass(classNumber) {
  selectedClass = classNumber;

  selectedClassDisplay.textContent = `Class ${classNumber}`;

  class0Button.classList.toggle("active", classNumber === 0);

  class1Button.classList.toggle("active", classNumber === 1);
}

class0Button.addEventListener("click", () => {
  if (isTraining) return;

  selectClass(0);
});

class1Button.addEventListener("click", () => {
  if (isTraining) return;

  selectClass(1);
});

/* =========================================================
   CANVAS CLICK
========================================================= */

boundaryCanvas.addEventListener("click", function (event) {
  if (isTraining) {
    return;
  }

  const rect = boundaryCanvas.getBoundingClientRect();

  const mouseX = event.clientX - rect.left;

  const mouseY = event.clientY - rect.top;

  const scaleX = boundaryCanvas.width / rect.width;

  const scaleY = boundaryCanvas.height / rect.height;

  const canvasX = mouseX * scaleX;

  const canvasY = mouseY * scaleY;

  const x1 = (canvasX - PADDING) / SCALE;

  const x2 = (boundaryCanvas.height - PADDING - canvasY) / SCALE;

  /*
   * Ignore clicks outside
   * the mathematical graph area.
   */

  if (x1 < X_MIN || x1 > X_MAX || x2 < Y_MIN || x2 > Y_MAX) {
    return;
  }

  /*
   * Add the new point.
   */

  trainingData.push({
    x: [Number(x1.toFixed(3)), Number(x2.toFixed(3))],

    y: selectedClass,
  });

  /*
   * Adding data invalidates
   * the previous model.
   */

  currentModel = createInitialModel();

  clearMetricsDisplay();

  drawCleanDataset();

  updateMathDisplay();
});

/* =========================================================
   DRAW DATA ONLY
========================================================= */

function drawCleanDataset() {
  ctx.clearRect(0, 0, boundaryCanvas.width, boundaryCanvas.height);

  drawDataPoints(trainingData);
}

/* =========================================================
   CLEAR METRICS
========================================================= */

function clearMetricsDisplay() {
  setText("weight1", "--");

  setText("weight2", "--");

  setText("bias", "--");

  setText("accuracy", "--");

  setText("testAccuracy", "--");

  setText("precision", "--");

  setText("recall", "--");

  setText("truePositive", "--");

  setText("trueNegative", "--");

  setText("falsePositive", "--");

  setText("falseNegative", "--");

  setText("epochDisplay", "0");

  setText("liveLoss", "--");

  setText("trainingState", "Ready");

  /*
   * Mathematical section.
   */

  setText("mathScore", "z = --");

  setText("mathProbability", "--");

  setText("mathPrediction", "--");

  setText("mathLoss", "--");

  setText("mathEpoch", "0");
}

/* =========================================================
   SMALL DOM HELPER
========================================================= */

function setText(id, value) {
  const element = document.getElementById(id);

  if (element) {
    element.textContent = value;
  }
}

/* =========================================================
   UPDATE MODEL PARAMETERS
========================================================= */

function updateParameterDisplay(model) {
  setText("weight1", model.weights[0].toFixed(4));

  setText("weight2", model.weights[1].toFixed(4));

  setText("bias", model.bias.toFixed(4));
}

/* =========================================================
   FIND REPRESENTATIVE POINT
========================================================= */

function getRepresentativePoint() {
  if (trainingData.length === 0) {
    return null;
  }

  /*
   * Prefer the most recently
   * added point.
   */

  return trainingData[trainingData.length - 1];
}

/* =========================================================
   UPDATE MATHEMATICAL EXPLANATION
========================================================= */

function updateMathDisplay() {
  const point = getRepresentativePoint();

  if (!point) {
    setText("mathX1", "--");

    setText("mathX2", "--");

    setText("mathClass", "--");

    return;
  }

  const x1 = point.x[0];

  const x2 = point.x[1];

  setText("mathX1", x1.toFixed(2));

  setText("mathX2", x2.toFixed(2));

  setText("mathClass", point.y);

  /*
   * Before training,
   * the model is zero-initialized.
   */

  const probability = predict(point.x, currentModel.weights, currentModel.bias);

  const z =
    currentModel.weights[0] * x1 +
    currentModel.weights[1] * x2 +
    currentModel.bias;

  setText("mathScore", `z = ${z.toFixed(4)}`);

  setText("mathProbability", `${(probability * 100).toFixed(1)}%`);

  setText("mathPrediction", probability >= 0.5 ? "Class 1" : "Class 0");

  const pointLoss = binaryCrossEntropy(point.y, probability);

  setText("mathLoss", pointLoss.toFixed(4));

  setText("mathEpoch", currentModel.currentEpoch);
}

/* =========================================================
   UPDATE FINAL METRICS
========================================================= */

function updateFinalMetrics(model) {
  const accuracy = calculateAccuracy(trainingData, model.weights, model.bias);

  const testAccuracy = calculateAccuracy(testData, model.weights, model.bias);

  const confusionMatrix = calculateConfusionMatrix(
    trainingData,
    model.weights,
    model.bias,
  );

  const metrics = calculateMetrics(confusionMatrix);

  setText("accuracy", `${(accuracy * 100).toFixed(1)}%`);

  setText("testAccuracy", `${(testAccuracy * 100).toFixed(1)}%`);

  setText("precision", `${(metrics.precision * 100).toFixed(1)}%`);

  setText("recall", `${(metrics.recall * 100).toFixed(1)}%`);

  setText("truePositive", confusionMatrix.TP);

  setText("trueNegative", confusionMatrix.TN);

  setText("falsePositive", confusionMatrix.FP);

  setText("falseNegative", confusionMatrix.FN);
}

/* =========================================================
   REDRAW MODEL
========================================================= */

function redrawTrainingState() {
  drawVisualization(trainingData, currentModel.weights, currentModel.bias);

  drawLossGraph(currentModel.lossHistory);

  updateParameterDisplay(currentModel);

  updateMathDisplay();

  setText("epochDisplay", currentModel.currentEpoch);

  if (currentModel.lossHistory.length > 0) {
    const loss = currentModel.lossHistory[currentModel.lossHistory.length - 1];

    setText("liveLoss", loss.toFixed(4));
  }
}

/* =========================================================
   TRAIN BUTTON
========================================================= */

trainButton.addEventListener("click", function () {
  if (isTraining) {
    return;
  }

  if (trainingData.length === 0) {
    alert("Add some training points first.");

    return;
  }

  const learningRate = Number(learningRateInput.value);

  const totalEpochs = Number(epochsInput.value);

  if (!Number.isFinite(learningRate) || learningRate <= 0) {
    alert("Learning rate must be greater than 0.");

    return;
  }

  if (!Number.isFinite(totalEpochs) || totalEpochs < 1) {
    alert("Epochs must be at least 1.");

    return;
  }

  /*
   * Start from zero weights.
   */

  currentModel = createInitialModel();

  clearMetricsDisplay();

  isTraining = true;

  setTrainingControls(true);

  setText("trainingState", "Learning");

  /*
   * Start browser-synchronized
   * animation.
   */

  animationId = requestAnimationFrame(() => {
    runTrainingFrame(learningRate, totalEpochs);
  });
});

/* =========================================================
   TRAINING FRAME
========================================================= */

function runTrainingFrame(learningRate, totalEpochs) {
  if (!isTraining) {
    return;
  }

  const epochsThisFrame = getEpochsPerFrame();

  for (let i = 0; i < epochsThisFrame; i++) {
    if (currentModel.currentEpoch >= totalEpochs) {
      finishTraining();

      return;
    }

    trainOneEpoch(trainingData, currentModel, learningRate);
  }

  /*
   * Redraw after the batch of
   * visible learning steps.
   */

  redrawTrainingState();

  trainButton.innerHTML = `<span>Epoch ${currentModel.currentEpoch} / ${totalEpochs}</span>
         <span class="button-arrow">→</span>`;

  if (currentModel.currentEpoch >= totalEpochs) {
    finishTraining();

    return;
  }

  animationId = requestAnimationFrame(() => {
    runTrainingFrame(learningRate, totalEpochs);
  });
}

/* =========================================================
   FINISH TRAINING
========================================================= */

function finishTraining() {
  if (animationId !== null) {
    cancelAnimationFrame(animationId);

    animationId = null;
  }

  isTraining = false;

  setTrainingControls(false);

  trainButton.innerHTML = `<span>Train Model</span>
         <span class="button-arrow">→</span>`;

  setText("trainingState", "Complete");

  redrawTrainingState();

  updateFinalMetrics(currentModel);
}

/* =========================================================
   CONTROL LOCKING
========================================================= */

function setTrainingControls(locked) {
  trainButton.disabled = locked;

  class0Button.disabled = locked;

  class1Button.disabled = locked;

  clearDataButton.disabled = locked;

  resetModelButton.disabled = locked;

  learningRateInput.disabled = locked;

  epochsInput.disabled = locked;

  trainingSpeedSelect.disabled = locked;
}

/* =========================================================
   RESET MODEL
========================================================= */

resetModelButton.addEventListener("click", function () {
  if (isTraining) {
    return;
  }

  currentModel = createInitialModel();

  clearMetricsDisplay();

  clearVisualization();

  drawCleanDataset();

  updateMathDisplay();
});

/* =========================================================
   CLEAR / RESTORE DATA
========================================================= */

clearDataButton.addEventListener("click", function () {
  if (isTraining) {
    return;
  }

  /*
   * Restore original six points.
   */

  trainingData.length = 0;

  originalTrainingData.forEach((sample) => {
    trainingData.push({
      x: [...sample.x],

      y: sample.y,
    });
  });

  currentModel = createInitialModel();

  clearMetricsDisplay();

  clearVisualization();

  drawCleanDataset();

  selectClass(0);

  updateMathDisplay();
});

/* =========================================================
   CLEAR VISUALIZATION
========================================================= */

function clearVisualization() {
  ctx.clearRect(0, 0, boundaryCanvas.width, boundaryCanvas.height);

  const lossCanvas = document.getElementById("lossCanvas");

  if (lossCanvas) {
    const lossCtx = lossCanvas.getContext("2d");

    lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
  }
}

/* =========================================================
   INITIALIZE
========================================================= */

selectClass(0);

clearMetricsDisplay();

drawCleanDataset();

updateMathDisplay();
