# Generating Synthetic Data with WGAN-GP

This repository hosts a robust implementation of a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** designed to generate high-fidelity synthetic tabular data. The project demonstrates an end-to-end pipeline from raw data preprocessing to advanced model evaluation.

## 🚀 Project Overview

The core of this project is the `wgan_gp.ipynb` notebook, which provides a detailed walkthrough of generating synthetic telecommunications data (CDRs) that preserves the statistical properties of the original dataset.

### Key Features
*   **Advanced GAN Architecture**: Utilizes WGAN-GP (Wasserstein GAN with Gradient Penalty) to overcome common GAN training instability and mode collapse issues.
*   **Custom Training Loop**: Implements a fine-tuned training loop with a configurable Critic-to-Generator update ratio (`n_critic`) and learning rate decay for optimal convergence.
*   **Comprehensive Evaluation**: Goes beyond visual inspection by calculating rigorous statistical metrics to quantify data quality.

## 📊 Methodology

The project workflow consists of the following stages:

1.  **Data Preprocessing**:
    *   Loading and cleaning the `sms-call-internet-mi-2013-11-01.csv` dataset.
    *   Handling missing values and normalizing features to ensure stable training dynamics.
    *   Filtering zero-variance rows to focus on meaningful data patterns.

2.  **Exploratory Data Analysis (EDA)**:
    *   Statistical profiling of the input data.
    *   Correlation analysis using **Spearman** correlation matrices and heatmaps.
    *   Distribution visualization via pair plots and histograms.

3.  **WGAN-GP Implementation**:
    *   **Generator**: A neural network designed to map random noise to the data space.
    *   **Critic (Discriminator)**: A network trained to approximate the Wasserstein distance between real and synthetic distributions.
    *   **Gradient Penalty**: Enforced to satisfy the 1-Lipschitz constraint, ensuring stable gradients throughout training.

4.  **Training Process**:
    *   The model is trained over multiple epochs with a dynamic learning rate.
    *   The Critic is updated multiple times for every Generator update to maintain a meaningful gradient signal.
    *   **Loss Tracking**: real-time monitoring of D_loss, G_loss, and Gradient Penalty values.

## 📈 Evaluation & Results

The quality of the generated synthetic data is evaluated using a suite of quantitative metrics:

*   **Fidelity (MSE)**: Measures how closely the synthetic data distribution matches the real data.
*   **Cosine Similarity**: Evaluates the directional alignment of feature vectors.
*   **KL Divergence**: Quantifies the information loss when approximating the real distribution with the synthetic one.
*   **Diversity**: Assesses the variance within the generated data to ensure the model isn't simply memorizing samples.
*   **Coverage**: Measures the spread of synthetic data across the real data manifold.

### Visual Validation
The notebook includes plotting functions to overlay Real vs. Synthetic data distributions (e.g., SMS In/Out, Call In/Out, Internet Usage) for visual inspection of feature matching.

## 📂 Repository Structure

*   `wgan_gp.ipynb`: The primary notebook containing all code, from data loading to evaluation.
*   `GANS.mp4`: Supplementary video demonstration.
*   `WGAN-GP.pdf` / `pptx`: Theoretical presentation slides and documentation.
*   `sms-call-internet-mi-2013-11-01.csv`: The source dataset.

## 🛠 Requirements

To reproduce these results, you will need a Python environment with:
*   **TensorFlow** (2.x)
*   **Pandas** & **NumPy** (Data manipulation)
*   **Matplotlib** & **Seaborn** (Visualization)

## 🚀 Usage

1.  Clone this repository:
    ```bash
    git clone https://github.com/FilippeZ/synthetic-data-generation-with-gans.git
    cd synthetic-data-generation-with-gans
    ```
2.  Install dependencies (ensure TensorFlow is configured for your hardware).
3.  Run the `wgan_gp.ipynb` notebook in Jupyter or Google Colab.

## 👤 Author

**Filippos Paraskevas Zygouris**
