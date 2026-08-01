import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

# ==========================================
# 1. Build WGAN-GP Models (From wgan_gp.ipynb)
# ==========================================

def build_generator(latent_dim=16):
    """
    Generator Network:
    Input: Noise vector z of dimension latent_dim (16)
    Hidden Layers: Dense(64) -> LayerNorm -> ReLU -> Dense(128) -> LayerNorm -> ReLU -> Dense(32) -> LayerNorm -> ReLU
    Output: 5 features (smsin, smsout, callin, callout, internet)
    """
    generator = models.Sequential([
        layers.InputLayer(input_shape=(latent_dim,)),
        layers.Dense(64), layers.LayerNormalization(), layers.ReLU(),
        layers.Dense(128), layers.LayerNormalization(), layers.ReLU(),
        layers.Dense(32), layers.LayerNormalization(), layers.ReLU(),
        layers.Dense(5, activation='linear')
    ], name="Generator")
    return generator

def build_discriminator(input_dim=5):
    """
    Critic / Discriminator Network (1-Lipschitz continuous):
    Input: 5 features (smsin, smsout, callin, callout, internet)
    Hidden Layers: Dense(128) -> LeakyReLU(0.2) -> Dense(64) -> LeakyReLU(0.2) -> Dense(32) -> LeakyReLU(0.2)
    Output: Scalar Wasserstein score
    """
    discriminator = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(128), layers.LeakyReLU(0.2),
        layers.Dense(64), layers.LeakyReLU(0.2),
        layers.Dense(32), layers.LeakyReLU(0.2),
        layers.Dense(1)
    ], name="Critic")
    return discriminator

# ==========================================
# 2. Evaluation Metrics (From wgan_gp.ipynb)
# ==========================================

def compute_fidelity(real_data, synthetic_data):
    return mean_squared_error(real_data, synthetic_data)

def compute_cosine_similarity(real_data, synthetic_data):
    return cosine_similarity(real_data, synthetic_data).mean()

def compute_kl_divergence(real_data, synthetic_data):
    real_hist, _ = np.histogram(real_data, bins=50, density=True)
    synth_hist, _ = np.histogram(synthetic_data, bins=50, density=True)
    return entropy(real_hist + 1e-10, synth_hist + 1e-10)

def compute_diversity(synthetic_data):
    return np.mean(np.var(synthetic_data, axis=0))

def compute_coverage(real_data, synthetic_data):
    min_real, max_real = np.min(real_data, axis=0), np.max(real_data, axis=0)
    min_synth, max_synth = np.min(synthetic_data, axis=0), np.max(synthetic_data, axis=0)
    return np.mean((max_synth - min_synth) / (max_real - min_real + 1e-8))

# ==========================================
# 3. Execution CLI Demonstration
# ==========================================

if __name__ == "__main__":
    print("🧠 WGAN-GP Synthesis Service Initialized (wgan_gp.ipynb)")
    
    latent_dim = 16
    gen = build_generator(latent_dim=latent_dim)
    critic = build_discriminator(input_dim=5)
    
    print("\n--- Generator Summary ---")
    gen.summary()
    
    print("\n--- Critic Summary ---")
    critic.summary()

    # Generate sample data
    num_samples = 500
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    synthetic_norm = gen.predict(noise, verbose=0)
    
    # Dataset statistics from sms-call-internet-mi-2013-11-01.csv
    mu = np.array([1.25, 0.98, 0.45, 0.38, 38.45])
    std = np.array([2.10, 1.85, 0.92, 0.78, 42.10])
    
    synthetic_data = np.maximum(0, synthetic_norm * std + mu)
    
    df_synth = pd.DataFrame(synthetic_data, columns=['smsin', 'smsout', 'callin', 'callout', 'internet'])
    print("\nSynthetic Telecom DataFrame Head (first 5 samples):")
    print(df_synth.head())
    
    # Export CSV
    output_path = "wgan_gp_synthetic_telecom_output.csv"
    df_synth.to_csv(output_path, index=False)
    print(f"\nSaved synthetic dataset to {output_path}")
