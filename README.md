# BoundaryLab

### Interactive Logistic Regression from Scratch

BoundaryLab is a browser-based visualization of **binary logistic regression implemented from scratch in JavaScript**.

It demonstrates how a model learns a linear decision boundary through **gradient descent**, while making the underlying mathematics visible through interactive data points, probability regions, loss, model parameters, and evaluation metrics.

## What It Demonstrates

- Interactive Class 0 / Class 1 data creation
- Logistic regression implemented without an ML library
- Sigmoid activation
- Binary cross-entropy loss
- Gradient calculation
- Gradient descent
- Learning rate and epochs
- Dynamic probability regions
- Linear decision boundary
- Training and test evaluation
- Accuracy, precision, and recall
- Confusion matrix
- Loss vs. epoch visualization
- Visualization of misclassified points

## Mathematical Core

The model follows the pipeline:

```text
Linear Score
    ↓
z = w₁x₁ + w₂x₂ + b
    ↓
Sigmoid
    ↓
p = 1 / (1 + e⁻ᶻ)
    ↓
Binary Cross-Entropy
    ↓
Gradient Calculation
    ↓
Gradient Descent
    ↓
Updated Weights & Bias
```

The decision boundary is defined by:

```text
w₁x₁ + w₂x₂ + b = 0
```

Training minimizes binary cross-entropy by repeatedly updating the model parameters using gradient descent.

## Project Structure

```text
boundary-lab/
├── index.html
├── style.css
├── README.md
├── .gitignore
└── js/
    ├── app.js
    ├── training.js
    └── visualization.js
```

- **`app.js`** — application state, user interaction, training control, and metrics
- **`training.js`** — logistic regression, sigmoid, loss, gradients, gradient descent, and evaluation
- **`visualization.js`** — data points, probability map, decision boundary, axes, and loss graph
- **`index.html`** — application structure
- **`style.css`** — interface styling

## Technologies

- HTML5
- CSS3
- JavaScript
- Canvas API
- No machine-learning framework

## Run Locally

Clone the repository and open `index.html` in a modern browser.

```bash
git clone <repository-url>
cd boundary-lab
```

No backend or package installation is required.

## Key Concept

Logistic regression produces a **linear decision boundary**. Therefore, datasets that cannot be separated linearly may contain misclassified points even after training.

This makes BoundaryLab useful for observing not only how optimization works, but also the limitations of a linear classification model.
