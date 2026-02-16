# Generating Synthetic Data with GANs

This repository contains a comprehensive implementation of Wasserstein GAN with Gradient Penalty (WGAN-GP) for generating synthetic tabular data. The project explores data preprocessing, model training, and evaluation of synthetic data quality.

## Project Overview

The main notebook `wgan_gp.ipynb` covers the following steps:

1.  **Data Loading & Preprocessing**: Loading the dataset `sms-call-internet-mi-2013-11-01.csv`, handling missing values, and normalization.
2.  **Exploratory Data Analysis (EDA)**: Visualizing data distributions and correlations (Spearman correlation, Heatmaps, Pair Plots).
3.  **WGAN-GP Implementation**: Implementing the Generator and Critic (Discriminator) networks using TensorFlow/Keras, along with the gradient penalty loss function.
4.  **Training**: Training the WGAN-GP model to learn the data distribution.
5.  **Synthetic Data Generation**: Generating new synthetic samples using the trained generator.
6.  **Evaluation**:
    *   **Fidelity**: Comparing the distribution of real vs. synthetic data (MSE).
    *   **Diversity**: Assessing the variety of generated samples.
    *   **Visualization**: Comparing histograms and scatter plots of real and synthetic data.

## Files

*   `wgan_gp.ipynb`: The main Jupyter Notebook containing the code and analysis.
*   `list_notebook_content.py`: A helper script to extract content from the notebook.
*   `GANS.mp4`: A video demonstration/recording related to the project.
*   `WGAN-GP.pdf` & `WGAN-GP.pptx`: Presentation materials explaining the theoretical background and project details.
*   `sms-call-internet-mi-2013-11-01.csv`: The dataset used for training.

## Requirements

*   Python 3.x
*   TensorFlow
*   Pandas
*   NumPy
*   Matplotlib
*   Seaborn

## Usage

1.  Clone the repository.
2.  Install the required dependencies.
3.  Run the `wgan_gp.ipynb` notebook to reproduce the results.

## Author

Filippos Paraskevas Zygouris
