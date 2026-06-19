<div align="center">
  <img src="assets/banner.svg" alt="Awesome Gradient Descent Banner" width="100%" />
</div>

<div align="center">

# 🚀 Awesome Gradient Descent

### A curated, comprehensive guide to gradient descent variants in Artificial Intelligence & Machine Learning

[![Awesome](https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github)](https://github.com/ishandutta2007/Awesome-Awesome-Awesome)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/ishandutta2007/Awesome-Gradient-Descent/pulls)
[![GitHub Stars](https://img.shields.io/github/stars/ishandutta2007/Awesome-Gradient-Descent?style=flat-square&logo=github&color=gold)](https://github.com/ishandutta2007/Awesome-Gradient-Descent/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/ishandutta2007/Awesome-Gradient-Descent?style=flat-square&logo=github)](https://github.com/ishandutta2007/Awesome-Gradient-Descent/network/members)
[![Variants](https://img.shields.io/badge/Variants-11-informational?style=flat-square&logo=tensorflow)](https://github.com/ishandutta2007/Awesome-Gradient-Descent)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Optimization-blue?style=flat-square&logo=pytorch)](https://github.com/ishandutta2007/Awesome-Gradient-Descent)
[![Research](https://img.shields.io/badge/Research-Papers%20Linked-orange?style=flat-square&logo=arxiv)](https://github.com/ishandutta2007/Awesome-Gradient-Descent)
<a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>

</div>

---

## 📖 About

**Gradient Descent** is the cornerstone optimization algorithm used to train AI and machine learning models. By iteratively minimising a cost function, it nudges model parameters in the direction of steepest descent. Depending on **data-processing methods**, **adaptation strategies**, and **architectural constraints**, several distinct variants have emerged over decades of research.

This repository catalogues **11 key variants**, providing:
- 📅 **Year of first introduction** and original paper citations
- 📐 **Mathematical formulas** and intuitive explanations
- 🗺️ **Mermaid flow diagrams** for each algorithm
- ✅ **Pros & Cons** and guidance on when to use each variant
- 🔗 **Links to detailed documentation pages**

---

## 🗂️ Table of Contents

- [📦 1. Data-Batching Variants](#1--data-batching-variants)
- [⚡ 2. Momentum & Adaptive Learning Rate Variants](#2--momentum--adaptive-learning-rate-variants)
- [🛡️ 3. Regularized & Constraints Variants](#3--regularized--constraints-variants)
- [⭐ Star History](#-star-history)

---

## 1. 📦 Data-Batching Variants

> These variants define **how much training data** the algorithm processes before updating the model's weights and biases. The choice of batch size affects convergence speed, stability, and memory usage.

| Variant | Year | Key Characteristics | First Paper |
|---|:---:|---|---|
| 📦 [**Batch Gradient Descent (Vanilla)**](docs/batch-gradient-descent.md) | 1847 | Computes the gradient relative to the **entire dataset**. *Pros:* Stable convergence and deterministic steps. *Cons:* Extremely slow and memory-intensive for large datasets. | Cauchy, A.-L. (1847). [*Méthode générale pour la résolution des systèmes d'équations simultanées*](https://gallica.bnf.fr/ark:/12148/bpt6k2982c/f540.item) |
| ⚡ [**Stochastic Gradient Descent (SGD)**](docs/stochastic-gradient-descent.md) | 1951 | Updates parameters using only **one randomly selected** training example at a time. *Pros:* Very fast and can escape local minima. *Cons:* Convergence is erratic and noisy. | Robbins, H. & Monro, S. (1951). [*A Stochastic Approximation Method*](https://doi.org/10.1214/aoms/1177729586). *Ann. Math. Statist.* |
| 🗂️ [**Mini-Batch Gradient Descent**](docs/mini-batch-gradient-descent.md) | 1998 | Splits the dataset into **small batches** (e.g., 32, 64, or 256 samples). *Pros:* Reduces variance while leveraging vectorised matrix math. *Cons:* Requires tuning the batch-size hyperparameter. | LeCun, Y. et al. (1998). [*Efficient BackProp*](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf). In *Neural Networks: Tricks of the Trade*. |

---

## 2. ⚡ Momentum & Adaptive Learning Rate Variants

> These advanced variations **dynamically adjust step sizes** to speed up convergence and navigate complex, non-convex loss landscapes. They range from classical momentum methods to per-parameter adaptive rate algorithms.

| Variant | Year | Key Characteristics | First Paper |
|---|:---:|---|---|
| 🔵 [**Momentum**](docs/momentum.md) | 1964 | Accelerates gradient descent by adding a **fraction of the previous update vector**. *Analogy:* Acts like a ball rolling down a hill, gaining kinetic energy over time. | Polyak, B. T. (1964). [*Some methods of speeding up the convergence of iteration methods*](https://doi.org/10.1016/0041-5553(64)90137-5). *USSR Comput. Math. Math. Phys.* |
| 🔭 [**Nesterov Accelerated Gradient (NAG)**](docs/nesterov-accelerated-gradient.md) | 1983 | Calculates the gradient based on **approximated future positions** (look-ahead). *Pros:* Prevents momentum from overshooting by slowing down before a valley bottom. | Nesterov, Y. (1983). [*A method for solving the convex programming problem with convergence rate O(1/k²)*](https://www.semanticscholar.org/paper/A-method-for-solving-the-convex-programming-problem-Nesterov/573ea6f73e10d93b1b4c48f2c7a6e4f35b6e8e6). *Doklady Akademii Nauk SSSR*, 269(3). |
| 📈 [**Adagrad**](docs/adagrad.md) | 2011 | Adapts the learning rate individually to **every parameter** based on historical squared gradients. *Pros:* Larger updates for rare parameters. *Cons:* Learning rate monotonically decreases and eventually vanishes. | Duchi, J., Hazan, E. & Singer, Y. (2011). [*Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*](https://jmlr.org/papers/v12/duchi11a.html). *JMLR*, 12. |
| 📉 [**RMSprop**](docs/rmsprop.md) | 2012 | Modifies Adagrad by restricting gradient accumulation to a **moving exponential window**. *Pros:* Resolves the vanishing learning rate problem effectively. | Hinton, G., Srivastava, N. & Swersky, K. (2012). [*Neural Networks for Machine Learning — Lecture 6e*](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf). Coursera. |
| ⚡ [**Adam**](docs/adam.md) | 2015 | Combines **Momentum** (first moment) and **RMSprop** (second moment) with bias correction. *Status:* The industry-standard default optimizer for deep neural networks. | Kingma, D. P. & Ba, J. (2015). [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980). *ICLR 2015*. |

---

## 3. 🛡️ Regularized & Constraints Variants

> These versions **modify the optimization step** to prevent overfitting, enforce mathematical constraints, or handle non-smooth objective functions — making them essential for modern transformer training and scientific computing.

| Variant | Year | Key Characteristics | First Paper |
|---|:---:|---|---|
| ⚖️ [**Weight Decay (AdamW / SGDW)**](docs/weight-decay-adamw.md) | 2019 | Decouples the regularization penalty **directly from** the gradient update calculations. *Significance:* Essential for training modern transformer architectures properly without corrupting adaptive rates. | Loshchilov, I. & Hutter, F. (2019). [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101). *ICLR 2019*. |
| 🔧 [**Proximal Gradient Descent**](docs/proximal-gradient-descent.md) | 1979 | Used for optimizing **non-differentiable** cost functions (like L1 regularization / Lasso). *Mechanism:* Splits the problem into a smooth gradient step followed by a geometry-mapping proximal step. | Lions, P.-L. & Mercier, B. (1979). [*Splitting algorithms for the sum of two nonlinear operators*](https://doi.org/10.1137/0716071). *SIAM J. Numer. Anal.*, 16(6). |
| 🎯 [**Projected Gradient Descent (PGD)**](docs/projected-gradient-descent.md) | 1964 | Ensures parameters always stay within **restricted boundaries** during training iterations. *Mechanism:* Maps coordinates back inside a defined feasible set if an optimization step pushes parameters outside allowed limits. | Goldstein, A. A. (1964). [*Convex programming in Hilbert space*](https://doi.org/10.1090/S0002-9904-1964-11178-2). *Bull. Amer. Math. Soc.*, 70(5). |

---

## 🧠 Quick Comparison

| Optimizer | Adaptive Rates | Momentum | Handles Constraints | Typical Use Case |
|---|:---:|:---:|:---:|---|
| Batch GD | ❌ | ❌ | ❌ | Small datasets, convex problems |
| SGD | ❌ | ❌ | ❌ | Online learning, very large datasets |
| Mini-Batch GD | ❌ | ❌ | ❌ | Standard deep learning baseline |
| Momentum | ❌ | ✅ | ❌ | Speeding up SGD on ill-conditioned problems |
| NAG | ❌ | ✅ | ❌ | RNNs, convex optimization with proofs |
| Adagrad | ✅ | ❌ | ❌ | Sparse NLP, short training runs |
| RMSprop | ✅ | ❌ | ❌ | RNNs, non-stationary objectives |
| Adam | ✅ | ✅ | ❌ | **Default for all deep learning** |
| AdamW | ✅ | ✅ | ❌ | **Transformers, LLMs** |
| Proximal GD | ❌ | ❌ | ✅ (soft) | Lasso, sparse recovery |
| PGD | ❌ | ❌ | ✅ (hard) | Adversarial training, physical systems |

---

## 📚 Core Resources

### 📖 Foundational Papers
- 📄 [A Stochastic Approximation Method](https://doi.org/10.1214/aoms/1177729586) — Robbins & Monro, 1951
- 📄 [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) — LeCun et al., 1998
- 📄 [Adaptive Subgradient Methods](https://jmlr.org/papers/v12/duchi11a.html) — Duchi, Hazan & Singer, 2011
- 📄 [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba, 2015
- 📄 [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, 2019

### 🎓 Learning Resources
- 🎥 [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/) — Sebastian Ruder's excellent survey
- 🎥 [CS231n: Neural Networks — Optimization](https://cs231n.github.io/neural-networks-3/) — Stanford
- 📘 [Deep Learning Book — Chapter 8](https://www.deeplearningbook.org/contents/optimization.html) — Goodfellow, Bengio & Courville
- 🎬 [Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks) — Geoffrey Hinton, Coursera

---

## 🤝 Contributing

Contributions are welcome! If you know of a gradient descent variant not listed here, or want to improve an existing page:

1. 🍴 Fork this repository
2. 🌿 Create a feature branch (`git checkout -b feature/new-variant`)
3. ✏️ Add your changes following the existing format
4. 📬 Open a Pull Request

Please ensure each new variant page includes a Mermaid diagram, mathematical formulas, Pros/Cons, and a citation to the original paper.

---

## ⭐ Star History

<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2FAwesome-Gradient-Descent&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Gradient-Descent&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Gradient-Descent&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/Awesome-Gradient-Descent&type=date&legend=bottom-right" />
</picture>
</a>
</div>

---

<div align="center">

**Made with ❤️ for the AI/ML community**

[![GitHub followers](https://img.shields.io/github/followers/ishandutta2007?label=Follow%20%40ishandutta2007&style=social)](https://github.com/ishandutta2007)

</div>
