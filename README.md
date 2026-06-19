# Awesome-Gradient-Descent
## Gradient Descent Variants in Artificial Intelligence

Gradient Descent is the cornerstone optimization algorithm used to train AI and machine learning models. By minimizing a cost function, it updates model parameters iteratively. Depending on data processing methods, adaptation strategies, and architectural constraints, several distinct variants exist.

---

## 1. Data-Batching Variants

These variants define how much training data the algorithm processes before updating the model's weights and biases.

*   **Batch Gradient Descent (Vanilla)**
    *   Computes the gradient of the cost function relative to the entire dataset.
    *   *Pros:* Stable convergence trajectories and highly deterministic steps.
    *   *Cons:* Extremely slow and memory-intensive for large datasets.
*   **Stochastic Gradient Descent (SGD)**
    *   Updates parameters sequentially using only one randomly selected training example at a time.
    *   *Pros:* Very fast to compute and can escape local minima due to high variance.
    *   *Cons:* Frequent fluctuations make convergence highly erratic and noisy.
*   **Mini-Batch Gradient Descent**
    *   Strikes a balance by splitting the dataset into small, manageable batches (e.g., 32, 64, or 256 samples).
    *   *Pros:* Reduces update variance while leveraging highly optimized vectorised matrix math.
    *   *Cons:* Requires tuning an additional hyperparameter (the batch size).

---

## 2. Momentum & Adaptive Learning Rate Variants

These advanced variations dynamically adjust step sizes to speed up convergence and navigate complex loss landscapes.

*   **Momentum**
    *   Accelerates gradient descent by adding a fraction of the previous step's update vector.
    *   *Analogy:* Acts like a ball rolling down a hill, gaining kinetic energy over time.
*   **Nesterov Accelerated Gradient (NAG)**
    *   An anticipatory look-ahead mechanism that calculates the gradient based on approximated future positions.
    *   *Pros:* Prevents the momentum from overshooting down slopes by slowing down before a valley bottom.
*   **Adagrad (Adaptive Gradient Algorithm)**
    *   Adapts the learning rate individually to every parameter based on historical squared gradients.
    *   *Pros:* Performs larger updates for infrequent parameters and smaller updates for frequent ones.
    *   *Cons:* The learning rate monotonically decreases and eventually vanishes completely.
*   **RMSprop (Root Mean Squared Propagation)**
    *   Modifies Adagrad by restricting the historical gradient accumulation to a moving exponential window.
    *   *Pros:* Resolves the vanishing learning rate problem effectively.
*   **Adam (Adaptive Moment Estimation)**
    *   Combines the principles of both Momentum (first moment) and RMSprop (second moment).
    *   *Status:* The industry-standard default optimizer for deep neural networks.

---

## 3. Regularized & Constraints Variants

These versions modify the optimization step to prevent overfitting or satisfy physical system parameters.

*   **Weight Decay (AdamW / SGDW)**
    *   Decouples the regularization penalty directly from the gradient update calculations.
    *   *Significance:* Essential for training modern transformer architectures properly without corrupting adaptive rates.
*   **Proximal Gradient Descent**
    *   Used for optimizing non-differentiable cost functions (like L1 regularization / Lasso).
    *   *Mechanism:* Splits the problem into a smooth gradient step followed by a geometry-mapping proximal step.
*   **Projected Gradient Descent (PGD)**
    *   Ensures parameters always stay within restricted boundaries during training iterations.
    *   *Mechanism:* Map coordinates back inside a defined feasible set if an optimization step pushes parameters outside allowed limits.
