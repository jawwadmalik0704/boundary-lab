/* =========================================================
   BOUNDARYLAB VISUALIZATION
========================================================= */

/* =========================
   BOUNDARY CANVAS
========================= */

const boundaryCanvas = document.getElementById("boundaryCanvas");

const ctx = boundaryCanvas.getContext("2d");

/* =========================
   GRAPH SETTINGS
========================= */

const PADDING = 60;

const SCALE = 80;

const X_MIN = 0;

const X_MAX = 6;

const Y_MIN = 0;

const Y_MAX = 6;

/* =========================
   COORDINATE CONVERSION
========================= */

function canvasX(x) {
  return PADDING + x * SCALE;
}

function canvasY(y) {
  return boundaryCanvas.height - PADDING - y * SCALE;
}

/* =========================
   DRAW AXES
========================= */

function drawAxes() {
  ctx.save();

  /*
   * X axis
   */

  ctx.beginPath();

  ctx.moveTo(canvasX(X_MIN), canvasY(Y_MIN));

  ctx.lineTo(canvasX(X_MAX), canvasY(Y_MIN));

  /*
   * Y axis
   */

  ctx.moveTo(canvasX(X_MIN), canvasY(Y_MIN));

  ctx.lineTo(canvasX(X_MIN), canvasY(Y_MAX));

  ctx.strokeStyle = "#344054";

  ctx.lineWidth = 1.5;

  ctx.stroke();

  /*
   * Tick labels
   */

  ctx.fillStyle = "#667085";

  ctx.font = "12px Arial";

  ctx.textAlign = "center";

  for (let value = 0; value <= X_MAX; value++) {
    const x = canvasX(value);

    ctx.beginPath();

    ctx.moveTo(x, canvasY(0));

    ctx.lineTo(x, canvasY(0) + 6);

    ctx.strokeStyle = "#667085";

    ctx.lineWidth = 1;

    ctx.stroke();

    ctx.fillText(value.toString(), x, canvasY(0) + 22);
  }

  /*
   * Y ticks
   */

  ctx.textAlign = "right";

  for (let value = 0; value <= Y_MAX; value++) {
    const y = canvasY(value);

    ctx.beginPath();

    ctx.moveTo(canvasX(0) - 6, y);

    ctx.lineTo(canvasX(0), y);

    ctx.strokeStyle = "#667085";

    ctx.lineWidth = 1;

    ctx.stroke();

    ctx.fillText(value.toString(), canvasX(0) - 10, y + 4);
  }

  /*
   * X axis label
   */

  ctx.textAlign = "center";

  ctx.font = "bold 13px Arial";

  ctx.fillStyle = "#344054";

  ctx.fillText(
    "X₁",
    (canvasX(X_MIN) + canvasX(X_MAX)) / 2,
    boundaryCanvas.height - 10,
  );

  /*
   * Y axis label
   */

  ctx.save();

  ctx.translate(16, (canvasY(Y_MIN) + canvasY(Y_MAX)) / 2);

  ctx.rotate(-Math.PI / 2);

  ctx.fillText("X₂", 0, 0);

  ctx.restore();

  ctx.restore();
}

/* =========================
   PROBABILITY BACKGROUND
========================= */

function drawProbabilityMap(weights, bias) {
  const cellSize = 6;

  /*
   * Draw only inside
   * the graph area.
   */

  for (let px = canvasX(X_MIN); px < canvasX(X_MAX); px += cellSize) {
    for (let py = canvasY(Y_MAX); py < canvasY(Y_MIN); py += cellSize) {
      const x = (px - PADDING) / SCALE;

      const y = (boundaryCanvas.height - PADDING - py) / SCALE;

      const probability = predict([x, y], weights, bias);

      /*
       * Below 0.5:
       * blue region.
       *
       * Above 0.5:
       * red region.
       *
       * Near 0.5:
       * lighter transition.
       */

      if (probability < 0.5) {
        const strength = 0.07 + (0.5 - probability) * 0.18;

        ctx.fillStyle = `rgba(
                        30,
                        100,
                        255,
                        ${strength}
                    )`;
      } else {
        const strength = 0.07 + (probability - 0.5) * 0.18;

        ctx.fillStyle = `rgba(
                        255,
                        60,
                        60,
                        ${strength}
                    )`;
      }

      ctx.fillRect(px, py, cellSize, cellSize);
    }
  }
}

/* =========================
   DATA POINTS
========================= */

function drawDataPoints(data, weights = null, bias = null) {
  for (const sample of data) {
    const x = canvasX(sample.x[0]);

    const y = canvasY(sample.x[1]);

    let isCorrect = true;

    /*
     * Determine whether
     * the model classified
     * the point correctly.
     */

    if (weights !== null && bias !== null) {
      const probability = predict(sample.x, weights, bias);

      const predictedClass = probability >= 0.5 ? 1 : 0;

      isCorrect = predictedClass === sample.y;
    }

    ctx.beginPath();

    ctx.arc(x, y, 8, 0, Math.PI * 2);

    /*
     * Keep the REAL class color.
     */

    ctx.fillStyle = sample.y === 0 ? "blue" : "red";

    ctx.fill();

    /*
     * Normal border.
     */

    ctx.strokeStyle = "black";

    ctx.lineWidth = 1.2;

    ctx.stroke();

    /*
     * Misclassified:
     * yellow outline.
     */

    if (!isCorrect) {
      ctx.beginPath();

      ctx.arc(x, y, 12, 0, Math.PI * 2);

      ctx.strokeStyle = "#f2c94c";

      ctx.lineWidth = 3;

      ctx.stroke();
    }
  }
}

/* =========================
   DECISION BOUNDARY
========================= */

function drawDecisionBoundary(weights, bias) {
  const w1 = weights[0];

  const w2 = weights[1];

  /*
   * If model is still
   * completely untrained,
   * don't draw anything.
   */

  if (Math.abs(w1) < 0.000001 && Math.abs(w2) < 0.000001) {
    return;
  }

  const points = [];

  function addPoint(x, y) {
    if (x >= X_MIN && x <= X_MAX && y >= Y_MIN && y <= Y_MAX) {
      const alreadyExists = points.some(
        (point) =>
          Math.abs(point.x - x) < 0.000001 && Math.abs(point.y - y) < 0.000001,
      );

      if (!alreadyExists) {
        points.push({
          x,
          y,
        });
      }
    }
  }

  /*
   * Left edge:
   *
   * w1*x + w2*y + b = 0
   */

  if (Math.abs(w2) > 0.000001) {
    const y = -(w1 * X_MIN + bias) / w2;

    addPoint(X_MIN, y);
  }

  /*
   * Right edge.
   */

  if (Math.abs(w2) > 0.000001) {
    const y = -(w1 * X_MAX + bias) / w2;

    addPoint(X_MAX, y);
  }

  /*
   * Bottom edge.
   */

  if (Math.abs(w1) > 0.000001) {
    const x = -(w2 * Y_MIN + bias) / w1;

    addPoint(x, Y_MIN);
  }

  /*
   * Top edge.
   */

  if (Math.abs(w1) > 0.000001) {
    const x = -(w2 * Y_MAX + bias) / w1;

    addPoint(x, Y_MAX);
  }

  /*
   * Need at least
   * two points to draw
   * a line.
   */

  if (points.length < 2) {
    return;
  }

  /*
   * Draw the first two
   * valid boundary points.
   */

  const p1 = points[0];

  const p2 = points[1];

  ctx.beginPath();

  ctx.moveTo(canvasX(p1.x), canvasY(p1.y));

  ctx.lineTo(canvasX(p2.x), canvasY(p2.y));

  ctx.strokeStyle = "black";

  ctx.lineWidth = 3;

  ctx.stroke();
}

/* =========================
   COMPLETE BOUNDARY GRAPH
========================= */

function drawVisualization(data, weights, bias) {
  ctx.clearRect(0, 0, boundaryCanvas.width, boundaryCanvas.height);

  /*
   * Background.
   */

  ctx.fillStyle = "#ffffff";

  ctx.fillRect(0, 0, boundaryCanvas.width, boundaryCanvas.height);

  /*
   * Probability regions.
   */

  drawProbabilityMap(weights, bias);

  /*
   * Axes.
   */

  drawAxes();

  /*
   * Data points.
   */

  drawDataPoints(data, weights, bias);

  /*
   * Boundary must be
   * above the regions
   * and data.
   */

  drawDecisionBoundary(weights, bias);
}

/* =========================================================
   LOSS GRAPH
========================================================= */

function drawLossGraph(lossHistory) {
  const lossCanvas = document.getElementById("lossCanvas");

  const lossCtx = lossCanvas.getContext("2d");

  lossCtx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);

  /*
   * White background.
   */

  lossCtx.fillStyle = "#ffffff";

  lossCtx.fillRect(0, 0, lossCanvas.width, lossCanvas.height);

  if (!lossHistory || lossHistory.length < 2) {
    return;
  }

  const paddingLeft = 60;

  const paddingRight = 25;

  const paddingTop = 25;

  const paddingBottom = 55;

  const graphWidth = lossCanvas.width - paddingLeft - paddingRight;

  const graphHeight = lossCanvas.height - paddingTop - paddingBottom;

  /*
   * Loss range.
   */

  const maxLoss = Math.max(...lossHistory);

  const minLoss = Math.min(...lossHistory);

  const lossRange = Math.max(maxLoss - minLoss, 0.000001);

  /*
   * Small visual padding
   * around loss values.
   */

  const yMin = Math.max(0, minLoss - lossRange * 0.08);

  const yMax = maxLoss + lossRange * 0.08;

  const yRange = Math.max(yMax - yMin, 0.000001);

  /* =========================
       AXES
    ========================== */

  lossCtx.beginPath();

  /*
   * Y axis.
   */

  lossCtx.moveTo(paddingLeft, paddingTop);

  lossCtx.lineTo(paddingLeft, lossCanvas.height - paddingBottom);

  /*
   * X axis.
   */

  lossCtx.lineTo(
    lossCanvas.width - paddingRight,
    lossCanvas.height - paddingBottom,
  );

  lossCtx.strokeStyle = "#344054";

  lossCtx.lineWidth = 1.5;

  lossCtx.stroke();

  /* =========================
       X AXIS EPOCH TICKS
    ========================== */

  const totalEpochs = lossHistory.length;

  /*
   * Always show 0.
   */

  const epochTicks = [0];

  /*
   * Every 100 epochs.
   */

  for (let epoch = 100; epoch < totalEpochs; epoch += 100) {
    epochTicks.push(epoch);
  }

  /*
   * Always show final epoch.
   */

  if (!epochTicks.includes(totalEpochs)) {
    epochTicks.push(totalEpochs);
  }

  lossCtx.font = "12px Arial";

  lossCtx.fillStyle = "#667085";

  lossCtx.textAlign = "center";

  for (const epoch of epochTicks) {
    const x = paddingLeft + (epoch / totalEpochs) * graphWidth;

    /*
     * Tick.
     */

    lossCtx.beginPath();

    lossCtx.moveTo(x, lossCanvas.height - paddingBottom);

    lossCtx.lineTo(x, lossCanvas.height - paddingBottom + 6);

    lossCtx.strokeStyle = "#667085";

    lossCtx.lineWidth = 1;

    lossCtx.stroke();

    /*
     * Label.
     */

    lossCtx.fillText(
      epoch.toString(),
      x,
      lossCanvas.height - paddingBottom + 22,
    );
  }

  /* =========================
       Y AXIS LABELS
    ========================== */

  const yTickCount = 5;

  lossCtx.textAlign = "right";

  for (let i = 0; i <= yTickCount; i++) {
    const value = yMin + (i / yTickCount) * yRange;

    const y =
      paddingTop + graphHeight - ((value - yMin) / yRange) * graphHeight;

    /*
     * Tick.
     */

    lossCtx.beginPath();

    lossCtx.moveTo(paddingLeft - 6, y);

    lossCtx.lineTo(paddingLeft, y);

    lossCtx.strokeStyle = "#667085";

    lossCtx.lineWidth = 1;

    lossCtx.stroke();

    /*
     * Label.
     */

    lossCtx.fillText(value.toFixed(3), paddingLeft - 10, y + 4);
  }

  /* =========================
       X AXIS TITLE
    ========================== */

  lossCtx.textAlign = "center";

  lossCtx.font = "bold 13px Arial";

  lossCtx.fillStyle = "#344054";

  lossCtx.fillText(
    "Epoch",
    paddingLeft + graphWidth / 2,
    lossCanvas.height - 12,
  );

  /* =========================
       Y AXIS TITLE
    ========================== */

  lossCtx.save();

  lossCtx.translate(16, paddingTop + graphHeight / 2);

  lossCtx.rotate(-Math.PI / 2);

  lossCtx.fillText("Loss", 0, 0);

  lossCtx.restore();

  /* =========================
       LOSS CURVE
    ========================== */

  lossCtx.beginPath();

  for (let i = 0; i < lossHistory.length; i++) {
    const x = paddingLeft + (i / (totalEpochs - 1)) * graphWidth;

    const y =
      paddingTop +
      graphHeight -
      ((lossHistory[i] - yMin) / yRange) * graphHeight;

    if (i === 0) {
      lossCtx.moveTo(x, y);
    } else {
      lossCtx.lineTo(x, y);
    }
  }

  lossCtx.strokeStyle = "#111111";

  lossCtx.lineWidth = 2.5;

  lossCtx.stroke();

  /* =========================
       START / END POINTS
    ========================== */

  const firstX = paddingLeft;

  const firstY =
    paddingTop + graphHeight - ((lossHistory[0] - yMin) / yRange) * graphHeight;

  const lastX = paddingLeft + graphWidth;

  const lastY =
    paddingTop +
    graphHeight -
    ((lossHistory[lossHistory.length - 1] - yMin) / yRange) * graphHeight;

  lossCtx.beginPath();

  lossCtx.arc(firstX, firstY, 3.5, 0, Math.PI * 2);

  lossCtx.fillStyle = "#111111";

  lossCtx.fill();

  lossCtx.beginPath();

  lossCtx.arc(lastX, lastY, 3.5, 0, Math.PI * 2);

  lossCtx.fill();
}

/* =========================
   GLOBAL EXPORTS
========================= */

window.drawVisualization = drawVisualization;

window.drawDataPoints = drawDataPoints;

window.drawDecisionBoundary = drawDecisionBoundary;

window.drawLossGraph = drawLossGraph;
