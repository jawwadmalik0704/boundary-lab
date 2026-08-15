/* =========================================================
   BOUNDARYLAB
   LOGISTIC REGRESSION FROM SCRATCH
   LIVE GRADIENT DESCENT VERSION
========================================================= */

/* =========================
   SIGMOID
========================= */

function sigmoid(z) {
  if (z >= 0) {
    const expValue = Math.exp(-z);

    return 1 / (1 + expValue);
  } else {
    const expValue = Math.exp(z);

    return expValue / (1 + expValue);
  }
}

/* =========================
   PREDICTION
========================= */

function predict(x, weights, bias) {
  const z = weights[0] * x[0] + weights[1] * x[1] + bias;

  return sigmoid(z);
}

/* =========================
   BINARY CROSS ENTROPY
========================= */

function binaryCrossEntropy(y, prediction) {
  const epsilon = 1e-15;

  const p = Math.min(Math.max(prediction, epsilon), 1 - epsilon);

  return -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
}

/* =========================
   ORIGINAL TRAINING DATA
========================= */

const trainingData = [
  { x: [1, 1], y: 0 },
  { x: [1, 2], y: 0 },
  { x: [2, 1], y: 0 },

  { x: [4, 4], y: 1 },
  { x: [4, 5], y: 1 },
  { x: [5, 4], y: 1 },
];

/* =========================
   TEST DATA
========================= */

const testData = [
  { x: [1.5, 1.5], y: 0 },
  { x: [1.8, 1.2], y: 0 },

  { x: [4.2, 4.2], y: 1 },
  { x: [4.5, 4.0], y: 1 },
];

/* =========================
   CALCULATE GRADIENTS
========================= */

function calculateGradients(data, weights, bias) {
  let dw1 = 0;

  let dw2 = 0;

  let db = 0;

  const n = data.length;

  for (const sample of data) {
    const prediction = predict(sample.x, weights, bias);

    const error = prediction - sample.y;

    /*
     * Gradient for w1
     */

    dw1 += error * sample.x[0];

    /*
     * Gradient for w2
     */

    dw2 += error * sample.x[1];

    /*
     * Gradient for bias
     */

    db += error;
  }

  /*
   * Average gradient
   */

  return {
    dw1: dw1 / n,

    dw2: dw2 / n,

    db: db / n,
  };
}

/* =========================
   CALCULATE LOSS
========================= */

function calculateLoss(data, weights, bias) {
  if (data.length === 0) {
    return 0;
  }

  let totalLoss = 0;

  for (const sample of data) {
    const prediction = predict(sample.x, weights, bias);

    totalLoss += binaryCrossEntropy(sample.y, prediction);
  }

  return totalLoss / data.length;
}

/* =========================================================
   ONE GRADIENT-DESCENT EPOCH
========================================================= */

function trainOneEpoch(data, model, learningRate) {
  /*
   * Calculate current gradients.
   */

  const gradients = calculateGradients(data, model.weights, model.bias);

  /*
   * Gradient descent update:
   *
   * w := w - learningRate * gradient
   */

  model.weights[0] -= learningRate * gradients.dw1;

  model.weights[1] -= learningRate * gradients.dw2;

  model.bias -= learningRate * gradients.db;

  /*
   * Calculate loss AFTER
   * the parameter update.
   */

  const loss = calculateLoss(data, model.weights, model.bias);

  model.lossHistory.push(loss);

  model.currentEpoch++;

  return {
    loss,

    gradients,

    epoch: model.currentEpoch,
  };
}

/* =========================
   TRAIN MODEL INSTANTLY
========================= */

function trainModel(data, learningRate, epochs) {
  const model = createInitialModel();

  for (let epoch = 0; epoch < epochs; epoch++) {
    trainOneEpoch(data, model, learningRate);
  }

  return {
    weights: [...model.weights],

    bias: model.bias,

    lossHistory: [...model.lossHistory],
  };
}

/* =========================
   ACCURACY
========================= */

function calculateAccuracy(data, weights, bias) {
  if (data.length === 0) {
    return 0;
  }

  let correct = 0;

  for (const sample of data) {
    const probability = predict(sample.x, weights, bias);

    const predictedClass = probability >= 0.5 ? 1 : 0;

    if (predictedClass === sample.y) {
      correct++;
    }
  }

  return correct / data.length;
}

/* =========================
   CONFUSION MATRIX
========================= */

function calculateConfusionMatrix(data, weights, bias) {
  let TP = 0;

  let TN = 0;

  let FP = 0;

  let FN = 0;

  for (const sample of data) {
    const probability = predict(sample.x, weights, bias);

    const predictedClass = probability >= 0.5 ? 1 : 0;

    if (sample.y === 1 && predictedClass === 1) {
      TP++;
    } else if (sample.y === 0 && predictedClass === 0) {
      TN++;
    } else if (sample.y === 0 && predictedClass === 1) {
      FP++;
    } else if (sample.y === 1 && predictedClass === 0) {
      FN++;
    }
  }

  return {
    TP,

    TN,

    FP,

    FN,
  };
}

/* =========================
   METRICS
========================= */

function calculateMetrics(confusionMatrix) {
  const { TP, TN, FP, FN } = confusionMatrix;

  const total = TP + TN + FP + FN;

  const accuracy = total === 0 ? 0 : (TP + TN) / total;

  const precision = TP + FP === 0 ? 0 : TP / (TP + FP);

  const recall = TP + FN === 0 ? 0 : TP / (TP + FN);

  return {
    accuracy,

    precision,

    recall,
  };
}

/* =========================
   MODEL FACTORY
========================= */

function createInitialModel() {
  return {
    weights: [0, 0],

    bias: 0,

    lossHistory: [],

    currentEpoch: 0,
  };
}

/* =========================
   EXPORTS
========================= */

window.sigmoid = sigmoid;

window.predict = predict;

window.binaryCrossEntropy = binaryCrossEntropy;

window.calculateGradients = calculateGradients;

window.calculateLoss = calculateLoss;

window.trainOneEpoch = trainOneEpoch;

window.trainModel = trainModel;

window.calculateAccuracy = calculateAccuracy;

window.calculateConfusionMatrix = calculateConfusionMatrix;

window.calculateMetrics = calculateMetrics;

window.createInitialModel = createInitialModel;

window.trainingData = trainingData;

window.testData = testData;

