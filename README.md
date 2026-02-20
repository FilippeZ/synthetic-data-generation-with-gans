# 🧠 Generating Synthetic Data with WGAN-GP — High-Fidelity Data Simulation

[Bridging the gap between raw data privacy and high-quality analytics through stable, synthetic tabular data generation.]

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview
This repository hosts a robust implementation of a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** designed to generate high-fidelity synthetic tabular data. The project provides an end-to-end pipeline—from raw data preprocessing to advanced model evaluation—demonstrating the generation of synthetic telecommunications data (CDRs) that strictly preserves the statistical properties of the original dataset.

## 🎯 The Problem
Modern data-driven analytics in telecommunications and other sensitive domains often faces a critical bottleneck:
* **Data Privacy:** Strict regulations limit the sharing and usage of real user data (e.g., Call Detail Records).
* **Training Instability:** Traditional GANs suffer from vanishing gradients and mode collapse, failing to capture the full diversity of complex tabular datasets.
* **Evaluation Deficit:** Generating data isn't enough; lacking rigorous statistical metrics makes it impossible to quantify the true quality and safety of synthetic alternatives.

## ✅ The Solution
This project leverages the advanced WGAN-GP architecture to overcome common GAN training instability, converting sensitive datasets into robust, safe synthetic alternatives.

| Feature | Technical Approach | Objective |
| :--- | :--- | :--- |
| **Stable Training** | Wasserstein Distance & Gradient Penalty | Mitigate mode collapse & stabilize gradients |
| **Statistical Fidelity** | Custom Evaluation Metrics (MSE, KL Div) | Ensure synthetic data mirrors real distributions |
| **Data Diversity** | Cosine Similarity & Coverage Tracking | Prevent model memorization (overfitting) |

---

## 🏗️ Architecture & Workflow
The system follows a sequential pipeline to ensure high-quality data synthesis:

1. **Ingestion & Preprocessing Layer:** Loading `sms-call-internet-mi-2013-11-01.csv`, handling missing values, normalizing features, and filtering out zero-variance rows to ensure stable training dynamics.
2. **Exploratory Data Analysis (EDA):** Statistical profiling, Spearman correlation matrices, and distribution visualizations (pair plots, histograms).
3. **Model Layer (WGAN-GP):** 
   * **Generator:** A deep neural network mapping random noise to the data space manifold.
   * **Critic (Discriminator):** Approximates the Wasserstein distance between real and synthetic distributions.
4. **Training Optimization:** Enforcing a 1-Lipschitz constraint via Gradient Penalty. The model features a dynamic learning rate and a configurable Critic-to-Generator update ratio (`n_critic`) for optimal convergence.

## 📂 Project Structure
```text
synthetic-data-generation-with-gans/
├── wgan_gp.ipynb                  # 🧠 Main Notebook (Data loading to evaluation)
├── sms-call-internet-mi-2013-11-01.csv # 📊 Source Dataset (CDRs)
├── GANS.mp4                       # 🎥 Video demonstration
└── WGAN-GP.pdf / pptx             # 📋 Theoretical presentation slides
```

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/FilippeZ/synthetic-data-generation-with-gans.git
cd synthetic-data-generation-with-gans
pip install pandas numpy matplotlib seaborn tensorflow
```

### 2. Execution
Launch the primary notebook to begin training and evaluation:
```bash
jupyter notebook wgan_gp.ipynb
```
*(Alternatively, you can run the notebook directly in Google Colab).*

## 📈 Evaluation & Results
We rigorously quantify the quality of the generated synthetic data using a suite of statistical metrics:

* **Fidelity (MSE):** Measures how closely the synthetic data distribution matches the real data.
* **Cosine Similarity:** Evaluates the directional alignment of feature vectors.
* **KL Divergence:** Quantifies the information loss when approximating the real distribution with the synthetic one.
* **Diversity:** Assesses the variance within the generated data to ensure the model isn't memorizing specific samples.
* **Coverage:** Measures the spread of synthetic data across the real data manifold.

*Visual Validation:* The notebook plots Real vs. Synthetic data distributions (e.g., SMS In/Out, Call In/Out, Internet Usage) to visually verify feature matching.

## 🔬 Deep Dive: WGAN-GP Mechanics
Traditional GANs minimize the Jensen-Shannon divergence, which leads to vanishing gradients when the generated and real distributions do not overlap.

**Wasserstein GAN:** Minimizes the Earth Mover's (Wasserstein-1) distance, providing a smooth, meaningful gradient everywhere.

**Gradient Penalty:** Instead of weight clipping (which causes pathological behavior), WGAN-GP penalizes the norm of the critic's gradient with respect to its input. This naturally enforces the required 1-Lipschitz continuity:

> $L = \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})] - \mathbb{E}_{x \sim P_r}[D(x)] + \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(||\nabla_{\hat{x}} D(\hat{x})||_2 - 1)^2]$

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Deep Learning Framework:** TensorFlow (2.x)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## 📄 License
Licensed under the MIT License — see LICENSE for details.

## 👤 Author
**Filippos-Paraskevas Zygouris**
[GitHub Profile](https://github.com/FilippeZ)
