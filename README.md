# Awesome-Gradient-Descent
## Gradient Descent Variants in Artificial Intelligence

Gradient Descent is the cornerstone optimization algorithm used to train AI and machine learning models. By minimizing a cost function, it updates model parameters iteratively. Depending on data processing methods, adaptation strategies, and architectural constraints, several distinct variants exist.

---

## 1. Data-Batching Variants

These variants define how much training data the algorithm processes before updating the model's weights and biases.

| Variant | Year | Key Characteristics | First Paper |
|---|---|---|---|
| **Batch Gradient Descent (Vanilla)** | 1847 | Computes the gradient relative to the entire dataset. *Pros:* Stable convergence and deterministic steps. *Cons:* Extremely slow and memory-intensive for large datasets. | Cauchy, A.-L. (1847). [*Méthode générale pour la résolution des systèmes d'équations simultanées*](https://gallica.bnf.fr/ark:/12148/bpt6k2982c/f540.item) |
| **Stochastic Gradient Descent (SGD)** | 1951 | Updates parameters using only one randomly selected training example at a time. *Pros:* Very fast and can escape local minima. *Cons:* Convergence is erratic and noisy. | Robbins, H. & Monro, S. (1951). [*A Stochastic Approximation Method*](https://doi.org/10.1214/aoms/1177729586). *Ann. Math. Statist.* |
| **Mini-Batch Gradient Descent** | 1998 | Splits the dataset into small batches (e.g., 32, 64, or 256 samples). *Pros:* Reduces variance while leveraging vectorised matrix math. *Cons:* Requires tuning the batch-size hyperparameter. | LeCun, Y. et al. (1998). [*Efficient BackProp*](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf). In *Neural Networks: Tricks of the Trade*. |

---

## 2. Momentum & Adaptive Learning Rate Variants

These advanced variations dynamically adjust step sizes to speed up convergence and navigate complex loss landscapes.

| Variant | Year | Key Characteristics | First Paper |
|---|---|---|---|
| **Momentum** | 1964 | Accelerates gradient descent by adding a fraction of the previous update vector. *Analogy:* Acts like a ball rolling down a hill, gaining kinetic energy over time. | Polyak, B. T. (1964). [*Some methods of speeding up the convergence of iteration methods*](https://doi.org/10.1016/0041-5553(64)90137-5). *USSR Comput. Math. Math. Phys.* |
| **Nesterov Accelerated Gradient (NAG)** | 1983 | Calculates the gradient based on approximated future positions. *Pros:* Prevents momentum from overshooting by slowing down before a valley bottom. | Nesterov, Y. (1983). [*A method for solving the convex programming problem with convergence rate O(1/k²)*](https://www.semanticscholar.org/paper/A-method-for-solving-the-convex-programming-problem-Nesterov/573ea6f73e10d93b1b4c48f2c7a6e4f35b6e8e6). *Doklady Akademii Nauk SSSR*, 269(3). |
| **Adagrad** | 2011 | Adapts the learning rate individually to every parameter based on historical squared gradients. *Pros:* Larger updates for rare parameters. *Cons:* Learning rate monotonically decreases and eventually vanishes. | Duchi, J., Hazan, E. & Singer, Y. (2011). [*Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*](https://jmlr.org/papers/v12/duchi11a.html). *JMLR*, 12. |
| **RMSprop** | 2012 | Modifies Adagrad by restricting gradient accumulation to a moving exponential window. *Pros:* Resolves the vanishing learning rate problem effectively. | Hinton, G., Srivastava, N. & Swersky, K. (2012). [*Neural Networks for Machine Learning — Lecture 6e*](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf). Coursera. |
| **Adam** | 2015 | Combines Momentum (first moment) and RMSprop (second moment). *Status:* The industry-standard default optimizer for deep neural networks. | Kingma, D. P. & Ba, J. (2015). [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980). *ICLR 2015*. |

---

## 3. Regularized & Constraints Variants

These versions modify the optimization step to prevent overfitting or satisfy physical system parameters.

| Variant | Year | Key Characteristics | First Paper |
|---|---|---|---|
| **Weight Decay (AdamW / SGDW)** | 2019 | Decouples the regularization penalty from gradient update calculations. *Significance:* Essential for training modern transformer architectures without corrupting adaptive rates. | Loshchilov, I. & Hutter, F. (2019). [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101). *ICLR 2019*. |
| **Proximal Gradient Descent** | 1979 | Used for optimizing non-differentiable cost functions (e.g., L1 / Lasso). *Mechanism:* Splits the problem into a smooth gradient step followed by a proximal step. | Lions, P.-L. & Mercier, B. (1979). [*Splitting algorithms for the sum of two nonlinear operators*](https://doi.org/10.1137/0716071). *SIAM J. Numer. Anal.*, 16(6). |
| **Projected Gradient Descent (PGD)** | 1964 | Ensures parameters stay within restricted boundaries during training. *Mechanism:* Maps coordinates back inside a feasible set if an update pushes them outside allowed limits. | Goldstein, A. A. (1964). [*Convex programming in Hilbert space*](https://doi.org/10.1090/S0002-9904-1964-11178-2). *Bull. Amer. Math. Soc.*, 70(5). |
