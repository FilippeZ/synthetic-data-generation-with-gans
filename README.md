<p align="center">
  <img src="logo.jpeg" alt="Synthesis Hub Logo" width="220" style="border-radius:16px;"/>
</p>

<h1 align="center">Synthesis Hub — WGAN-GP Platform</h1>
<h3 align="center">5G Healthcare IoT & Brain Tumor MRI Synthetic Data Generation</h3>

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/WGAN--GP-Gradient%20Penalty-8F4E00"/>
  <img src="https://img.shields.io/badge/Dataset-7200%20Brain%20MRIs-C62828"/>
  <img src="https://img.shields.io/badge/License-MIT-green"/>
</p>

---

## Overview

**Synthesis Hub** is an end-to-end **Generative AI** platform for the Healthcare sector built on the **WGAN-GP** (Wasserstein GAN with Gradient Penalty) architecture. It solves two major challenges in modern telemedicine:

| Pillar | Challenge Solved |
|--------|-----------------|
| **Section 1 — 5G Healthcare IoT** | Secure synthesis of telemedicine network traffic without exposing patient communication patterns |
| **Section 2 — Brain Tumor MRI** | Generation of realistic synthetic brain MRI scans per tumor class for data augmentation with 100% patient privacy |

The platform is delivered as a **full-stack web application** (Flask + HTML/CSS/JS) connected to two Jupyter notebooks covering the complete ML pipeline: preprocessing → EDA → training → synthesis → evaluation.

---

## Table of Contents

- [Architecture](#architecture)
- [Section 1 — 5G Healthcare IoT](#section-1--5g-healthcare-iot--anomaly-detection)
- [Section 2 — Brain Tumor MRI WGAN-GP](#section-2--brain-tumor-mri-wgan-gp)
- [Jupyter Notebooks](#jupyter-notebooks)
- [Web Application](#web-application)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [References](#references)

---

## Architecture

### WGAN-GP Core Principle

The Wasserstein GAN with Gradient Penalty enforces the **1-Lipschitz constraint** through a differentiable gradient penalty term, replacing weight clipping:

$$\mathcal{L}_{WGAN-GP} = \underbrace{\mathbb{E}_{x \sim P_r}[D(x)]}_{\text{real score}} - \underbrace{\mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})]}_{\text{fake score}} + \underbrace{\lambda \cdot \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]}_{\text{gradient penalty}}$$

Where $\hat{x} = \alpha x_{\text{real}} + (1-\alpha) x_{\text{fake}}$, $\alpha \sim U[0,1]$, and $\lambda = 10$.

---

## Section 1 — 5G Healthcare IoT & Anomaly Detection

### Dataset
Milan 5G Telecommunication Activity Dataset (`sms-call-internet-mi-2013-11-01.csv`):

| Feature | Description |
|---------|-------------|
| `smsin` | SMS messages received |
| `smsout` | SMS messages sent |
| `callin` | Voice calls received |
| `callout` | Voice calls made |
| `internet` | Internet data traffic (MB) |

### Tabular WGAN-GP Architecture

**Generator** (Dense + LayerNormalization):
```
z ∈ ℝ^16  →  Dense(64)  →  LayerNorm  →  ReLU
          →  Dense(128) →  LayerNorm  →  ReLU
          →  Dense(32)  →  LayerNorm  →  ReLU
          →  Dense(5)   →  Linear output  →  [smsin, smsout, callin, callout, internet]
```

**Critic** (Dense + LeakyReLU):
```
x ∈ ℝ^5  →  Dense(128) →  LeakyReLU(0.2)
         →  Dense(64)  →  LeakyReLU(0.2)
         →  Dense(32)  →  LeakyReLU(0.2)
         →  Dense(1)   →  Wasserstein score
```

### 5G Anomaly Detection
The trained Critic's Wasserstein score $\mathcal{W}(x)$ is used as an anomaly detector. Traffic patterns that deviate significantly from the learned distribution (e.g. DDoS surges) receive anomalous critic scores, triggering alerts.

---

## Section 2 — Brain Tumor MRI WGAN-GP

### Dataset
[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 7,200 Human Brain MRI Images:

| Class | Training | Testing | Description |
|-------|----------|---------|-------------|
| `glioma` | 1,400 | 400 | Malignant brain tumor originating from glial cells |
| `meningioma` | 1,400 | 400 | Tumor arising from meninges surrounding brain |
| `pituitary` | 1,400 | 400 | Tumor in the pituitary gland |
| `notumor` | 1,400 | 400 | Healthy brain — no tumor present |
| **Total** | **5,600** | **1,600** | **7,200 images** |

### WGAN-GP Deconvolutional Architecture

**Generator** (translated from user's Lasagne/Theano `build_net(nz=200)` to TF 2.x):

```python
# nz = 200 (latent vector dimension)
z ∈ ℝ^200
  → Dense(1024 × 4 × 4)                                   # Project
  → Reshape(4, 4, 1024)                                    # 4×4×1024
  → Conv2DTranspose(512, 4×4, stride=2) + BatchNorm + ReLU # 8×8×512
  → Conv2DTranspose(256, 4×4, stride=2) + BatchNorm + ReLU # 16×16×256
  → Conv2DTranspose(128, 4×4, stride=2) + BatchNorm + ReLU # 32×32×128
  → Conv2DTranspose(  1, 4×4, stride=2) + Sigmoid          # 64×64×1
```

**Critic / Discriminator**:

```python
x ∈ ℝ^{64×64×1}
  → Conv2D(128,  5×5, stride=2) + BatchNorm + LeakyReLU(0.2) # 32×32×128
  → Conv2D(256,  5×5, stride=2) + BatchNorm + LeakyReLU(0.2) # 16×16×256
  → Conv2D(512,  5×5, stride=2) + BatchNorm + LeakyReLU(0.2) # 8×8×512
  → Conv2D(1024, 5×5, stride=2) + BatchNorm + LeakyReLU(0.2) # 4×4×1024
  → Flatten → Dense(1)                                        # Wasserstein score
```

### Training Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Latent dim $n_z$ | 200 | User Lasagne code |
| Gradient penalty $\lambda$ | 10 | User Lasagne code |
| Critic steps per gen step | 5 | WGAN-GP paper |
| Learning rate | 5×10⁻⁵ | User Lasagne code |
| Adam $\beta_1$ | 0.5 | User Lasagne code |
| Adam $\beta_2$ | 0.9 | User Lasagne code |
| Batch size | 32 | Standard |
| Image resolution | 64×64×1 (grayscale) | User code |

### Output — 10×10 Montage
The notebook generates a `create_montage()` grid of 100 synthetic MRI images (matching the user's original Lasagne/Theano `create_montage(image)` function):

```
┌──────────────────────────────────┐
│  10 × 10 = 100 Synthetic MRIs   │
│  640 × 640 px output image       │
│  Saved: montages/final_{class}.png │
└──────────────────────────────────┘
```

---

## Jupyter Notebooks

### `wgan_gp.ipynb` — Section 1: 5G Healthcare IoT

Complete pipeline for 5G network traffic synthesis:

| Cell Group | Description |
|-----------|-------------|
| Data Loading | Load CSV, datetime parsing, chunk reading for 1M+ rows |
| Preprocessing | NaN filling, normalization (z-score: $x' = \frac{x - \mu}{\sigma}$) |
| EDA | Distribution plots, Spearman correlation heatmap, pair plots |
| WGAN-GP Model | Generator + Critic definitions, gradient penalty |
| Training | Full training loop with loss curves |
| Synthesis | Generate 2,500+ synthetic records, export CSV |
| Evaluation | `compute_fidelity`, `compute_cosine_similarity`, `compute_kl_divergence`, `compute_diversity`, `compute_coverage` |

### `wgan_gp_brain_tumor_mri.ipynb` — Section 2: Brain Tumor MRI

Full MRI synthesis pipeline (52 cells):

| Section | Description |
|---------|-------------|
| 1. Imports | TF 2.x, PIL, sklearn, scipy |
| 2. Config | All hyperparameters in one place |
| 3. EDA | Dataset registry, class distribution, real MRI samples grid, pixel histograms, mean MRI per class |
| 4. Preprocessing | `load_class_images()` → 64×64 grayscale [0,1], `tf.data.Dataset` pipeline |
| 5. Architecture | Generator & Critic with `.summary()`, Adam optimizers |
| 6. Training | Gradient penalty, `@tf.function` train steps, full loop, montage every 10 epochs, weight saving |
| 7. Loss Curves | G loss, D loss, Gradient Penalty plots |
| 8. Comparison | Real vs Synthetic side-by-side 8-column grid |
| 9. Evaluation | MSE, Cosine Similarity, KL Divergence, Diversity, Coverage, FID approximation |
| 10. Generation | 10×10 montage, export 50 individual images, latent space interpolation |
| 11. Inference | Load saved weights and generate without retraining |

---

## Web Application

### Stack
- **Backend:** Python Flask 3.x + TensorFlow 2.20
- **Frontend:** HTML5 + Vanilla CSS + Tailwind CDN + Chart.js
- **Design:** Espresso & Cream theme (Primary: `#271310`, Accent: `#FF8F00`)
- **Typography:** Playfair Display (headers) + Inter (body)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Server health & model info |
| `POST` | `/api/synthesize/telecom` | Generate synthetic 5G traffic records |
| `POST` | `/api/detect_anomalies` | WGAN-GP critic-based DDoS detection |
| `POST` | `/api/mri/real_samples` | Fetch real MRI images from dataset |
| `POST` | `/api/synthesize/dcgan_mri` | Generate synthetic brain MRI images |
| `POST` | `/api/mri/montage` | Generate 10×10 montage (100 MRIs) |
| `GET` | `/api/mri/montage_download` | Download montage PNG |
| `POST` | `/api/mri/train` | Start background WGAN-GP training |
| `GET` | `/api/mri/train_status` | Poll training progress & loss history |
| `POST` | `/api/mri/load_weights` | Load pre-trained weights for a class |
| `GET` | `/api/download/csv` | Export synthetic telecom data as CSV |
| `POST` | `/api/upload` | Upload custom 5G telemetry CSV |

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/FilippeZ/synthetic-data-generation-with-gans.git
cd synthetic-data-generation-with-gans
```

### 2. Install Dependencies
```bash
pip install tensorflow flask pandas numpy scikit-learn scipy pillow matplotlib seaborn jupyter
```

### 3. Download the Datasets

**Brain Tumor MRI Dataset** — Place in project root:
```
Training/
  glioma/        (1,400 .jpg images)
  meningioma/    (1,400 .jpg images)
  notumor/       (1,400 .jpg images)
  pituitary/     (1,400 .jpg images)
Testing/
  glioma/        (400 .jpg images)
  meningioma/    (400 .jpg images)
  notumor/       (400 .jpg images)
  pituitary/     (400 .jpg images)
```
> Download from: [Kaggle — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

**5G Milan Telecom Dataset** — Place in project root:
```
sms-call-internet-mi-2013-11-01.csv
```
> Download from: [Telecom Italia Big Data Challenge](https://dandelion.eu/datamine/open-big-data/)

---

## Usage

### Step 1 — Run the MRI Notebook (Training)
```bash
jupyter notebook wgan_gp_brain_tumor_mri.ipynb
```
1. Set `TARGET_CLASS = "glioma"` in Cell 2
2. Run all cells (Kernel → Restart & Run All)
3. Weights saved to `saved_weights/gen_glioma.weights.h5`
4. Repeat for all 4 classes

### Step 2 — Run the 5G IoT Notebook
```bash
jupyter notebook wgan_gp.ipynb
```

### Step 3 — Start the Web Application
```bash
python app.py
```
Open: **http://localhost:5000**

### Web App Tabs

| Tab | Functionality |
|-----|--------------|
| **Sec 1: 5G Healthcare IoT** | Generate synthetic telemetry, adjust sample count, view loss curves, export CSV |
| **5G Threat & DDoS** | Simulate DDoS attack, see Wasserstein critic scores, view 24h traffic baseline |
| **Sec 2: Brain Tumor MRI** | Select tumor class, view real dataset MRIs, generate synthetic MRIs, 10×10 montage |
| **Fidelity & Diversity Metrics** | MSE, Cosine Similarity, KL Divergence, Diversity, Coverage bar charts |
| **WGAN-GP Code Models** | Architecture code reference |

---

## Evaluation Metrics

All metrics are implemented in both notebooks and the Flask API:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Fidelity MSE** | $\frac{1}{n}\sum(x_r - x_s)^2$ | Lower = more faithful |
| **Cosine Similarity** | $\frac{x_r \cdot x_s}{\|x_r\|\|x_s\|}$ | Higher = more similar distribution |
| **KL Divergence** | $\sum P(x)\log\frac{P(x)}{Q(x)}$ | Lower = better distribution match |
| **Diversity (Var)** | $\mathbb{E}[\text{Var}(x_s)]$ | Higher = more diverse outputs |
| **Coverage Ratio** | $\frac{\max(x_s)-\min(x_s)}{\max(x_r)-\min(x_r)}$ | Higher = better manifold coverage |
| **FID Approximation** | $(\mu_r-\mu_s)^2 + (\sigma_r-\sigma_s)^2$ | Lower = better visual quality |

---

## Project Structure

```
synthetic-data-generation-with-gans/
│
├── 📓 wgan_gp.ipynb                    # Section 1: 5G IoT full pipeline
├── 📓 wgan_gp_brain_tumor_mri.ipynb    # Section 2: Brain Tumor MRI full pipeline
│
├── 🌐 index.html                       # Web application frontend
├── 🐍 app.py                           # Flask backend server
├── 📦 package.json                     # npm start script
│
├── 🖼️  logo.jpeg                        # Project logo
├── 📄 WGAN-GP.pdf                      # Research paper reference
│
├── 📁 stitch_synthesis_iot_medical_hub/ # UI design assets
│   ├── 5g_iot_anomaly_detection/
│   └── medical_data_synthesizer/
│
├── 📁 saved_weights/                   # Generated by notebook (gitignored)
│   ├── gen_glioma.weights.h5
│   ├── gen_meningioma.weights.h5
│   ├── gen_notumor.weights.h5
│   └── gen_pituitary.weights.h5
│
├── 📁 Training/                        # Brain Tumor MRI dataset (gitignored)
│   ├── glioma/   (1,400 images)
│   ├── meningioma/ (1,400 images)
│   ├── notumor/  (1,400 images)
│   └── pituitary/ (1,400 images)
│
├── 📁 Testing/                         # Brain Tumor MRI test set (gitignored)
│   ├── glioma/   (400 images)
│   ├── meningioma/ (400 images)
│   ├── notumor/  (400 images)
│   └── pituitary/ (400 images)
│
└── 📁 montages/                        # Generated montages (gitignored)
```

---

## Dataset

### Brain Tumor MRI Dataset
- **Source:** [Kaggle — Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Total Images:** 7,200 human brain MRI scans
- **Format:** JPEG, 512×512 px (resized to 64×64 for training)
- **Classes:** Glioma, Meningioma, Pituitary, No Tumor
- **License:** Public domain for research use

### 5G Telecom Dataset (Milan)
- **Source:** [Telecom Italia Big Data Challenge](https://dandelion.eu/datamine/open-big-data/)
- **Records:** 1,891,928 rows of timestamped network activity
- **Features:** smsin, smsout, callin, callout, internet
- **Coverage:** Milan city grid, November 2013

---

## References

1. **Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).** Improved Training of Wasserstein GANs. *NeurIPS 2017.* [arXiv:1704.03490](https://arxiv.org/abs/1704.03490)

2. **Arjovsky, M., Chintala, S., & Bottou, L. (2017).** Wasserstein GAN. *ICML 2017.* [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)

3. **Goodfellow, I., et al. (2014).** Generative Adversarial Nets. *NeurIPS 2014.*

4. **Radford, A., Metz, L., & Chintala, S. (2015).** Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)

5. **Brain Tumor MRI Dataset.** Masoud Nickparvar, Kaggle 2021.

6. **Telecom Italia Big Data Challenge Dataset.** Barlacchi, G., et al. A multi-source dataset of urban life in the city of Milan and the Province of Trentino. *Scientific Data, 2015.*

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <img src="logo.jpeg" alt="Synthesis Hub" width="80" style="border-radius:8px; opacity:0.8;"/>
  <br/>
  <strong>Synthesis Hub</strong> — WGAN-GP Healthcare AI Platform<br/>
  <em>5G IoT · Brain Tumor MRI · Differential Privacy · Generative AI</em>
</p>
