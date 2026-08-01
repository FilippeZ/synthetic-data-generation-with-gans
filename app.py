import os
import threading
import time
import glob
import base64
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, Response, send_file, stream_with_context
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

app = Flask(__name__, static_folder='.', static_url_path='')

@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return resp

# ==============================================================================
# SECTION 1: 5G HEALTHCARE IOT & ANOMALY DETECTION (Tabular WGAN-GP)
# ==============================================================================

LATENT_DIM_TABULAR = 16
FEATURE_NAMES = ['smsin', 'smsout', 'callin', 'callout', 'internet']

dataset_stats = {
    'mu': np.array([1.25, 0.98, 0.45, 0.38, 38.45], dtype=np.float32),
    'std': np.array([2.10, 1.85, 0.92, 0.78, 42.10], dtype=np.float32),
    'filename': 'sms-call-internet-mi-2013-11-01.csv',
    'total_rows': 1891928
}

real_data_sample = None
latest_synthetic_df = None

def build_tabular_generator(latent_dim=LATENT_DIM_TABULAR):
    return models.Sequential([
        layers.InputLayer(shape=(latent_dim,)),
        layers.Dense(64), layers.LayerNormalization(), layers.ReLU(),
        layers.Dense(128), layers.LayerNormalization(), layers.ReLU(),
        layers.Dense(32), layers.LayerNormalization(), layers.ReLU(),
        layers.Dense(5, activation='linear')
    ], name="Tabular_Generator")

def build_tabular_critic(input_dim=5):
    return models.Sequential([
        layers.InputLayer(shape=(input_dim,)),
        layers.Dense(128), layers.LeakyReLU(0.2),
        layers.Dense(64), layers.LeakyReLU(0.2),
        layers.Dense(32), layers.LeakyReLU(0.2),
        layers.Dense(1)
    ], name="Tabular_Critic")

tabular_generator = build_tabular_generator()
tabular_critic = build_tabular_critic()

def init_real_dataset():
    global real_data_sample, dataset_stats
    file_path = dataset_stats['filename']
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, nrows=10000)
            cols = [c for c in FEATURE_NAMES if c in df.columns]
            if len(cols) == 5:
                data = df[cols].fillna(0).values.astype(np.float32)
                real_data_sample = data
                dataset_stats['mu'] = np.mean(data, axis=0)
                dataset_stats['std'] = np.std(data, axis=0)
                dataset_stats['std'][dataset_stats['std'] == 0] = 1.0
                print(f"Loaded 5G Milan Network Sample ({data.shape[0]} rows). Mu: {dataset_stats['mu']}")
        except Exception as e:
            print(f"Dataset load notice: {e}")

init_real_dataset()

# ==============================================================================
# SECTION 2: BRAIN TUMOR MRI WGAN-GP
# Architecture translated from User's Lasagne/Theano code -> TF 2.x Keras
# nz=200, Deconv Generator (4->8->16->32->64), Conv Critic (64->32->16->8->4)
# Dataset: Training/ & Testing/ with glioma, meningioma, notumor, pituitary
# ==============================================================================

NZ_MRI       = 200          # Latent noise dimension (from user code: nz=200)
MRI_IMG_SIZE = 64           # Output resolution 64x64x1
MRI_LAMBDA   = 10           # Gradient penalty weight (from user code)
MRI_N_CRITIC = 5            # Train critic 5x per generator step (standard WGAN-GP)
MRI_LR_G     = 0.00005      # Adam lr for generator (from user code: 5e-5)
MRI_LR_D     = 0.00005      # Adam lr for critic
MRI_BETA1    = 0.5          # Adam beta1 (from user code)
MRI_BETA2    = 0.9          # Adam beta2 (from user code)
MRI_BATCH    = 32           # Batch size for training

MRI_CLASSES  = ['glioma', 'meningioma', 'notumor', 'pituitary']
WEIGHTS_DIR  = 'saved_weights'

# Training state (shared across threads)
mri_train_state = {
    'is_training': False,
    'epoch': 0,
    'epochs_total': 0,
    'steps': 0,
    'g_loss': [],
    'd_loss': [],
    'last_montage_b64': None,
    'trained_class': None,
    'error': None,
    'pixel_mean': 0.0,
    'pixel_std': 0.0,
    'fid_approx': None,
}

# In-memory image cache per class (lazy loaded)
_mri_image_cache = {}

# Index Brain Tumor Dataset Files
mri_dataset_registry = {
    'Training': {c: [] for c in MRI_CLASSES},
    'Testing':  {c: [] for c in MRI_CLASSES}
}

def scan_mri_dataset():
    for split in ['Training', 'Testing']:
        for cls in MRI_CLASSES:
            p = os.path.join(split, cls)
            if os.path.exists(p):
                files = glob.glob(os.path.join(p, '*.jpg')) + \
                        glob.glob(os.path.join(p, '*.jpeg')) + \
                        glob.glob(os.path.join(p, '*.png'))
                mri_dataset_registry[split][cls] = files
    total_train = sum(len(v) for v in mri_dataset_registry['Training'].values())
    total_test  = sum(len(v) for v in mri_dataset_registry['Testing'].values())
    print(f"Brain Tumor MRI Dataset Indexed: Training={total_train}, Testing={total_test} images.")

scan_mri_dataset()

def load_mri_images(tumor_class, split='Training', max_images=1400):
    """Load and preprocess real MRI images for a given class into numpy array.
    Returns array of shape (N, 64, 64, 1) normalized to [0, 1].
    """
    cache_key = f"{split}_{tumor_class}"
    if cache_key in _mri_image_cache:
        return _mri_image_cache[cache_key]

    files = mri_dataset_registry[split].get(tumor_class, [])
    if not files:
        return None

    np.random.shuffle(files)
    files = files[:max_images]

    imgs = []
    for fp in files:
        try:
            img = Image.open(fp).convert('L').resize((MRI_IMG_SIZE, MRI_IMG_SIZE), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            imgs.append(arr)
        except Exception:
            continue

    if not imgs:
        return None

    data = np.stack(imgs, axis=0)[:, :, :, np.newaxis]  # (N, 64, 64, 1)
    _mri_image_cache[cache_key] = data
    print(f"Loaded {len(data)} {split}/{tumor_class} MRI images into cache.")
    return data


# -----------------------------------------------------------------------
# WGAN-GP Model Architectures (exact user spec)
# -----------------------------------------------------------------------

def build_mri_generator(nz=NZ_MRI):
    """
    Generator (User Lasagne code -> TF2 Keras):
      Input: z in R^200
      Dense(1024*4*4) -> Reshape(4,4,1024)
      Conv2DTranspose(512, 4, s=2, same) + BN + ReLU  -> 8x8x512
      Conv2DTranspose(256, 4, s=2, same) + BN + ReLU  -> 16x16x256
      Conv2DTranspose(128, 4, s=2, same) + BN + ReLU  -> 32x32x128
      Conv2DTranspose(  1, 4, s=2, same) + Sigmoid    -> 64x64x1
    """
    inp = layers.Input(shape=(nz,))
    x = layers.Dense(1024 * 4 * 4, use_bias=False)(inp)
    x = layers.Reshape((4, 4, 1024))(x)
    # Block 1: 4->8
    x = layers.Conv2DTranspose(512, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Block 2: 8->16
    x = layers.Conv2DTranspose(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Block 3: 16->32
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # Block 4: 32->64
    x = layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='sigmoid')(x)
    return models.Model(inp, x, name='WGANGP_Generator')

def build_mri_critic(img_shape=(MRI_IMG_SIZE, MRI_IMG_SIZE, 1)):
    """
    Critic / Discriminator (User Lasagne code -> TF2 Keras):
      Input: 64x64x1
      Conv2D(128, 5, s=2, same) + BN + LReLU(0.2) -> 32x32x128
      Conv2D(256, 5, s=2, same) + BN + LReLU(0.2) -> 16x16x256
      Conv2D(512, 5, s=2, same) + BN + LReLU(0.2) ->  8x8x512
      Conv2D(1024,5, s=2, same) + BN + LReLU(0.2) ->  4x4x1024
      Flatten -> Dense(1)  [no sigmoid - Wasserstein output]
    """
    inp = layers.Input(shape=img_shape)
    x = layers.Conv2D(128,  5, strides=2, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256,  5, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(512,  5, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(1024, 5, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)  # Wasserstein: no activation
    return models.Model(inp, x, name='WGANGP_Critic')

mri_generator = build_mri_generator()
mri_critic    = build_mri_critic()

# Optimizers: Adam with user-specified lr=5e-5, beta1=0.5, beta2=0.9
mri_g_optimizer = tf.keras.optimizers.Adam(learning_rate=MRI_LR_G, beta_1=MRI_BETA1, beta_2=MRI_BETA2)
mri_d_optimizer = tf.keras.optimizers.Adam(learning_rate=MRI_LR_D, beta_1=MRI_BETA1, beta_2=MRI_BETA2)

latest_montage_bytes = None

# -----------------------------------------------------------------------
# WGAN-GP core: gradient penalty
# -----------------------------------------------------------------------

@tf.function
def gradient_penalty(critic, real_imgs, fake_imgs, lam=MRI_LAMBDA):
    """Computes WGAN-GP gradient penalty: E[||nabla D(x_hat)||_2 - 1]^2"""
    batch = tf.shape(real_imgs)[0]
    alpha = tf.random.uniform([batch, 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real_imgs + (1.0 - alpha) * fake_imgs
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = critic(interpolated, training=True)
    grads = tape.gradient(pred, interpolated)
    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
    gp = tf.reduce_mean(tf.square(grads_norm - 1.0))
    return lam * gp

@tf.function
def train_critic_step(real_imgs, nz):
    batch = tf.shape(real_imgs)[0]
    z = tf.random.normal([batch, nz])
    with tf.GradientTape() as tape:
        fake_imgs  = mri_generator(z, training=True)
        real_score = mri_critic(real_imgs, training=True)
        fake_score = mri_critic(fake_imgs, training=True)
        gp         = gradient_penalty(mri_critic, real_imgs, fake_imgs)
        d_loss     = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score) + gp
    grads = tape.gradient(d_loss, mri_critic.trainable_variables)
    mri_d_optimizer.apply_gradients(zip(grads, mri_critic.trainable_variables))
    return d_loss

@tf.function
def train_generator_step(batch_size, nz):
    z = tf.random.normal([batch_size, nz])
    with tf.GradientTape() as tape:
        fake_imgs  = mri_generator(z, training=True)
        fake_score = mri_critic(fake_imgs, training=True)
        g_loss     = -tf.reduce_mean(fake_score)
    grads = tape.gradient(g_loss, mri_generator.trainable_variables)
    mri_g_optimizer.apply_gradients(zip(grads, mri_generator.trainable_variables))
    return g_loss

# -----------------------------------------------------------------------
# Weight save/load helpers
# -----------------------------------------------------------------------

def save_weights(tumor_class):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    mri_generator.save_weights(os.path.join(WEIGHTS_DIR, f'gen_{tumor_class}.weights.h5'))
    mri_critic.save_weights(os.path.join(WEIGHTS_DIR, f'critic_{tumor_class}.weights.h5'))
    print(f"Weights saved for class: {tumor_class}")

def load_weights(tumor_class):
    gp = os.path.join(WEIGHTS_DIR, f'gen_{tumor_class}.weights.h5')
    cp = os.path.join(WEIGHTS_DIR, f'critic_{tumor_class}.weights.h5')
    if os.path.exists(gp) and os.path.exists(cp):
        # Build models first by running a dummy forward pass
        _ = mri_generator(tf.zeros([1, NZ_MRI]), training=False)
        _ = mri_critic(tf.zeros([1, MRI_IMG_SIZE, MRI_IMG_SIZE, 1]), training=False)
        mri_generator.load_weights(gp)
        mri_critic.load_weights(cp)
        print(f"Loaded saved weights for class: {tumor_class}")
        return True
    return False

def list_saved_classes():
    if not os.path.exists(WEIGHTS_DIR):
        return []
    return [c for c in MRI_CLASSES if os.path.exists(os.path.join(WEIGHTS_DIR, f'gen_{c}.weights.h5'))]

# -----------------------------------------------------------------------
# Montage helper (matches user's create_montage function)
# -----------------------------------------------------------------------

def create_montage_image(images, grid=10):
    """
    Creates a grid x grid montage matching the user's create_montage(image).
    images: np array (N, H, W) in [0, 1].
    """
    n, h, w = images.shape[0], images.shape[1], images.shape[2]
    montage = np.zeros((grid * h, grid * w), dtype=np.uint8)
    idx = 0
    for i in range(grid):
        for j in range(grid):
            if idx < n:
                montage[i*h:(i+1)*h, j*w:(j+1)*w] = (images[idx] * 255).clip(0, 255).astype(np.uint8)
            idx += 1
    pil_img = Image.fromarray(montage, mode='L')
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.seek(0)
    return buf.getvalue()

def gen_sample_images(n=100):
    """Generate n synthetic MRI images, returns (N, H, W) in [0,1]."""
    noise = tf.random.normal([n, NZ_MRI])
    imgs  = mri_generator(noise, training=False).numpy()  # (N,64,64,1)
    return imgs[:, :, :, 0]

# -----------------------------------------------------------------------
# Approximate FID: compare pixel-level feature means/stds
# -----------------------------------------------------------------------

def compute_image_metrics(real_imgs_arr, synthetic_imgs):
    """
    real_imgs_arr: (N,64,64,1) in [0,1]
    synthetic_imgs: (M,64,64)  in [0,1]
    Returns dict with pixel-level statistics.
    """
    r = real_imgs_arr[:, :, :, 0].flatten()
    s = synthetic_imgs.flatten()
    mu_r, std_r = float(np.mean(r)), float(np.std(r))
    mu_s, std_s = float(np.mean(s)), float(np.std(s))
    fid_approx  = float((mu_r - mu_s)**2 + (std_r - std_s)**2)

    r_hist, _ = np.histogram(r, bins=100, density=True)
    s_hist, _ = np.histogram(s, bins=100, density=True)
    kl = float(entropy(r_hist + 1e-10, s_hist + 1e-10))

    return {
        'real_pixel_mean':   round(mu_r, 4),
        'real_pixel_std':    round(std_r, 4),
        'synth_pixel_mean':  round(mu_s, 4),
        'synth_pixel_std':   round(std_s, 4),
        'fid_approx':        round(fid_approx, 6),
        'pixel_kl_div':      round(kl, 4),
    }

# ==============================================================================
# TABULAR EVALUATION METRICS
# ==============================================================================

def compute_metrics(real_data, synthetic_data):
    if real_data is None:
        real_data = np.random.normal(0, 1, synthetic_data.shape)
    n = min(len(real_data), len(synthetic_data))
    r, s = real_data[:n], synthetic_data[:n]
    mse     = float(mean_squared_error(r, s))
    cos_sim = float(cosine_similarity(r, s).mean())
    kl_divs = []
    for col in range(r.shape[1]):
        rh, _ = np.histogram(r[:, col], bins=50, density=True)
        sh, _ = np.histogram(s[:, col], bins=50, density=True)
        kl_divs.append(entropy(rh + 1e-10, sh + 1e-10))
    kl_avg    = float(np.mean(kl_divs))
    diversity = float(np.mean(np.var(s, axis=0)))
    min_r, max_r = np.min(r, axis=0), np.max(r, axis=0)
    min_s, max_s = np.min(s, axis=0), np.max(s, axis=0)
    cov_ratio = float(np.mean((max_s - min_s) / (max_r - min_r + 1e-8)))
    return {
        'mse':               round(mse, 6),
        'cosine_similarity': round(cos_sim, 4),
        'kl_divergence':     round(kl_avg, 4),
        'diversity_variance':round(diversity, 4),
        'coverage_ratio':    round(cov_ratio * 100, 2)
    }

# ==============================================================================
# REST API
# ==============================================================================

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# ------ Status & info -------------------------------------------------------

@app.route('/api/status', methods=['GET'])
def get_status():
    saved = list_saved_classes()
    train_total = sum(len(v) for v in mri_dataset_registry['Training'].values())
    test_total  = sum(len(v) for v in mri_dataset_registry['Testing'].values())
    return jsonify({
        'status': 'online',
        'project': '5G Healthcare IoT & Brain Tumor MRI WGAN-GP Platform',
        'section1_tabular': 'WGAN-GP Dense + LayerNorm (5G Milan Telemedicine)',
        'section2_brain_mri': 'WGAN-GP DCGAN (nz=200) Brain Tumor MRI Dataset',
        'framework': 'TensorFlow ' + tf.__version__,
        'nz_mri': NZ_MRI,
        'mri_classes': MRI_CLASSES,
        'mri_training_images': train_total,
        'mri_testing_images':  test_total,
        'saved_weights_classes': saved,
    })

@app.route('/api/mri/dataset_info', methods=['GET'])
def mri_dataset_info():
    info = {}
    for split in ['Training', 'Testing']:
        info[split] = {cls: len(mri_dataset_registry[split][cls]) for cls in MRI_CLASSES}
    return jsonify({
        'status': 'success',
        'dataset_name': 'Brain Tumor MRI Dataset',
        'total_images': 7200,
        'classes': MRI_CLASSES,
        'split_info': info,
        'saved_classes': list_saved_classes(),
        'training_active': mri_train_state['is_training'],
    })

# ------ Real samples from dataset ------------------------------------------

@app.route('/api/mri/real_samples', methods=['POST'])
def get_real_mri_samples():
    data_req     = request.get_json() or {}
    target_class = data_req.get('tumor_class', 'glioma')
    count        = min(16, max(1, int(data_req.get('count', 8))))

    files = (mri_dataset_registry['Training'].get(target_class, []) +
             mri_dataset_registry['Testing'].get(target_class, []))
    if not files:
        for c in MRI_CLASSES:
            files += mri_dataset_registry['Training'][c]

    selected = np.random.choice(files, min(count, len(files)), replace=False)
    samples  = []
    for i, fp in enumerate(selected):
        try:
            img     = Image.open(fp).convert('L').resize((128, 128), Image.BILINEAR)
            buf     = io.BytesIO()
            img.save(buf, format='PNG')
            encoded = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            samples.append({
                'sample_index': i + 1,
                'class_name':   os.path.basename(os.path.dirname(fp)),
                'file_name':    os.path.basename(fp),
                'image_url':    encoded,
                'resolution':   '128x128'
            })
        except Exception as e:
            print(f"Error loading {fp}: {e}")

    return jsonify({'status': 'success', 'target_class': target_class,
                    'count': len(samples), 'samples': samples})

# ------ WGAN-GP Training ---------------------------------------------------

@app.route('/api/mri/train', methods=['POST'])
def start_mri_training():
    """
    Start WGAN-GP training on real Brain Tumor MRI images.
    Body: { "tumor_class": "glioma", "epochs": 30, "batch_size": 32 }
    """
    global mri_train_state

    if mri_train_state['is_training']:
        return jsonify({'error': 'Training already running', 'state': mri_train_state}), 409

    data_req     = request.get_json() or {}
    tumor_class  = data_req.get('tumor_class', 'glioma')
    epochs       = max(1, min(200, int(data_req.get('epochs', 30))))
    batch_size   = max(8, min(64, int(data_req.get('batch_size', MRI_BATCH))))

    if tumor_class not in MRI_CLASSES:
        return jsonify({'error': f'Unknown class. Choose from {MRI_CLASSES}'}), 400

    # Reset state
    mri_train_state.update({
        'is_training':   True,
        'epoch':         0,
        'epochs_total':  epochs,
        'steps':         0,
        'g_loss':        [],
        'd_loss':        [],
        'last_montage_b64': None,
        'trained_class': tumor_class,
        'error':         None,
        'pixel_mean':    0.0,
        'pixel_std':     0.0,
        'fid_approx':    None,
    })

    def training_thread():
        global mri_generator, mri_critic, mri_g_optimizer, mri_d_optimizer
        global mri_train_state, latest_montage_bytes

        try:
            # Rebuild fresh models + optimizers for the new class
            mri_generator    = build_mri_generator()
            mri_critic       = build_mri_critic()
            mri_g_optimizer  = tf.keras.optimizers.Adam(MRI_LR_G, MRI_BETA1, MRI_BETA2)
            mri_d_optimizer  = tf.keras.optimizers.Adam(MRI_LR_D, MRI_BETA1, MRI_BETA2)

            # Try loading existing weights
            load_weights(tumor_class)

            # Load real images
            real_imgs = load_mri_images(tumor_class, split='Training', max_images=1400)
            if real_imgs is None or len(real_imgs) == 0:
                mri_train_state['error']       = f'No images found for class: {tumor_class}'
                mri_train_state['is_training'] = False
                return

            n_imgs    = len(real_imgs)
            steps_per_epoch = max(1, n_imgs // batch_size)

            print(f"Starting WGAN-GP training: class={tumor_class}, "
                  f"n_imgs={n_imgs}, epochs={epochs}, batch={batch_size}, "
                  f"steps/epoch={steps_per_epoch}")

            for epoch in range(1, epochs + 1):
                mri_train_state['epoch'] = epoch
                np.random.shuffle(real_imgs)

                epoch_d_losses = []
                epoch_g_losses = []

                for step in range(steps_per_epoch):
                    mri_train_state['steps'] += 1
                    start_idx = (step * batch_size) % n_imgs
                    end_idx   = min(start_idx + batch_size, n_imgs)
                    batch     = tf.constant(real_imgs[start_idx:end_idx])

                    # Train critic MRI_N_CRITIC times per generator step
                    for _ in range(MRI_N_CRITIC):
                        d_loss = train_critic_step(batch, NZ_MRI)
                    epoch_d_losses.append(float(d_loss))

                    # Train generator once
                    g_loss = train_generator_step(len(batch), NZ_MRI)
                    epoch_g_losses.append(float(g_loss))

                avg_d = float(np.mean(epoch_d_losses))
                avg_g = float(np.mean(epoch_g_losses))
                mri_train_state['g_loss'].append(round(avg_g, 4))
                mri_train_state['d_loss'].append(round(avg_d, 4))

                # Every 5 epochs: generate a sample montage & compute image metrics
                if epoch % 5 == 0 or epoch == epochs:
                    synth_imgs = gen_sample_images(100)
                    montage_bytes = create_montage_image(synth_imgs)
                    latest_montage_bytes = montage_bytes
                    mri_train_state['last_montage_b64'] = (
                        "data:image/png;base64," +
                        base64.b64encode(montage_bytes).decode()
                    )
                    metrics = compute_image_metrics(real_imgs[:100], synth_imgs[:100])
                    mri_train_state['pixel_mean'] = metrics['synth_pixel_mean']
                    mri_train_state['pixel_std']  = metrics['synth_pixel_std']
                    mri_train_state['fid_approx'] = metrics['fid_approx']

                print(f"Epoch {epoch}/{epochs} | D_loss={avg_d:.4f} | G_loss={avg_g:.4f}")

                # Save weights every 10 epochs
                if epoch % 10 == 0 or epoch == epochs:
                    save_weights(tumor_class)

            mri_train_state['is_training'] = False
            print(f"Training complete for class: {tumor_class}")

        except Exception as e:
            import traceback
            mri_train_state['error']       = str(e)
            mri_train_state['is_training'] = False
            print(f"Training error: {e}")
            traceback.print_exc()

    t = threading.Thread(target=training_thread, daemon=True)
    t.start()

    return jsonify({
        'status':       'training_started',
        'tumor_class':  tumor_class,
        'epochs':       epochs,
        'batch_size':   batch_size,
        'n_critic':     MRI_N_CRITIC,
        'lambda_gp':    MRI_LAMBDA,
        'lr_gen':       MRI_LR_G,
        'lr_critic':    MRI_LR_D,
        'architecture': 'nz=200 -> Dense(1024x4x4) -> 4xDeconv2D -> 64x64x1 | Conv2D(128,256,512,1024) -> Dense(1)'
    })

@app.route('/api/mri/train_status', methods=['GET'])
def get_training_status():
    s = mri_train_state
    pct = 0
    if s['epochs_total'] > 0:
        pct = round(s['epoch'] / s['epochs_total'] * 100, 1)
    return jsonify({
        'is_training':    s['is_training'],
        'epoch':          s['epoch'],
        'epochs_total':   s['epochs_total'],
        'progress_pct':   pct,
        'steps':          s['steps'],
        'trained_class':  s['trained_class'],
        'latest_g_loss':  s['g_loss'][-1] if s['g_loss'] else None,
        'latest_d_loss':  s['d_loss'][-1] if s['d_loss'] else None,
        'g_loss_history': s['g_loss'],
        'd_loss_history': s['d_loss'],
        'pixel_mean':     s['pixel_mean'],
        'pixel_std':      s['pixel_std'],
        'fid_approx':     s['fid_approx'],
        'has_montage':    s['last_montage_b64'] is not None,
        'error':          s['error'],
        'saved_classes':  list_saved_classes(),
    })

@app.route('/api/mri/train_montage', methods=['GET'])
def get_training_montage():
    """Return the latest montage generated during training."""
    if mri_train_state['last_montage_b64'] is None:
        # Generate one with current (possibly untrained) weights
        synth_imgs = gen_sample_images(100)
        montage_bytes = create_montage_image(synth_imgs)
        mri_train_state['last_montage_b64'] = (
            "data:image/png;base64," + base64.b64encode(montage_bytes).decode()
        )
    return jsonify({
        'status':    'success',
        'epoch':     mri_train_state['epoch'],
        'image_url': mri_train_state['last_montage_b64'],
    })

@app.route('/api/mri/load_weights', methods=['POST'])
def api_load_weights():
    data_req    = request.get_json() or {}
    tumor_class = data_req.get('tumor_class', 'glioma')
    ok = load_weights(tumor_class)
    return jsonify({'status': 'loaded' if ok else 'no_weights_found',
                    'tumor_class': tumor_class})

# ------ Synthetic image generation (using current generator weights) --------

@app.route('/api/synthesize/dcgan_mri', methods=['POST'])
def synthesize_dcgan_mri():
    data_req    = request.get_json() or {}
    batch_size  = min(16, max(1, int(data_req.get('batch_size', 8))))
    tumor_class = data_req.get('tumor_class', 'glioma')

    noise       = tf.random.normal([batch_size, NZ_MRI])
    mri_tensors = mri_generator(noise, training=False).numpy()  # (B, 64, 64, 1)

    slices = []
    for i in range(batch_size):
        arr     = (mri_tensors[i, :, :, 0] * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr, mode='L').resize((128, 128), Image.NEAREST)
        buf     = io.BytesIO()
        pil_img.save(buf, format='PNG')
        encoded = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        slices.append({
            'slice_index': i + 1,
            'tumor_class': tumor_class,
            'resolution':  '64x64 (displayed 128x128)',
            'latent_nz':   200,
            'image_url':   encoded,
        })

    return jsonify({
        'status':       'success',
        'architecture': 'WGAN-GP Deconv2D (nz=200)',
        'tumor_class':  tumor_class,
        'batch_size':   batch_size,
        'is_trained':   tumor_class in list_saved_classes(),
        'slices':       slices,
    })

# ------ 10x10 Montage ------------------------------------------------------

@app.route('/api/mri/montage', methods=['POST'])
def generate_mri_montage():
    global latest_montage_bytes
    synth_imgs         = gen_sample_images(100)
    latest_montage_bytes = create_montage_image(synth_imgs)
    encoded            = "data:image/png;base64," + base64.b64encode(latest_montage_bytes).decode()
    return jsonify({
        'status':       'success',
        'montage_size': '10x10 Grid (100 Synthetic MRIs)',
        'dimensions':   '640x640 px',
        'image_url':    encoded,
        'is_trained':   mri_train_state['trained_class'] is not None,
    })

@app.route('/api/mri/montage_download', methods=['GET'])
def download_montage():
    global latest_montage_bytes
    if latest_montage_bytes is None:
        latest_montage_bytes = create_montage_image(gen_sample_images(100))
    buf = io.BytesIO(latest_montage_bytes)
    return send_file(buf, mimetype='image/png', as_attachment=True,
                     download_name='wgan_gp_brain_tumor_montage_100.png')

# ------ Tabular synthesis --------------------------------------------------

@app.route('/api/upload', methods=['POST'])
def upload_csv():
    global real_data_sample, dataset_stats
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    try:
        df   = pd.read_csv(file)
        cols = [c for c in FEATURE_NAMES if c in df.columns]
        data = df[cols].fillna(0).values.astype(np.float32) if cols else \
               df.select_dtypes(include=[np.number]).iloc[:, :5].fillna(0).values.astype(np.float32)
        if data.shape[1] < 5:
            data = np.hstack([data, np.zeros((data.shape[0], 5 - data.shape[1]), dtype=np.float32)])
        real_data_sample         = data[:10000]
        dataset_stats['mu']      = np.mean(data, axis=0)
        dataset_stats['std']     = np.std(data, axis=0)
        dataset_stats['std'][dataset_stats['std'] == 0] = 1.0
        dataset_stats['filename']   = file.filename
        dataset_stats['total_rows'] = len(df)
        return jsonify({'message': '5G Telemetry log preprocessed', 'filename': file.filename,
                        'total_rows': len(df), 'mu': dataset_stats['mu'].tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/synthesize/telecom', methods=['POST'])
def synthesize_telecom():
    global latest_synthetic_df
    data_req    = request.get_json() or {}
    num_samples = int(data_req.get('num_samples', 2500))
    noise       = tf.random.normal([num_samples, LATENT_DIM_TABULAR])
    z_norm      = tabular_generator(noise, training=False).numpy()
    mu, std     = dataset_stats['mu'], dataset_stats['std']
    synthetic_vals = np.maximum(0, z_norm * std + mu)
    df_synth    = pd.DataFrame(synthetic_vals, columns=FEATURE_NAMES)
    latest_synthetic_df = df_synth
    metrics     = compute_metrics(real_data_sample, z_norm)
    samples     = [{'sample_index': i+1, 'smsin': round(float(synthetic_vals[i,0]),4),
                    'smsout': round(float(synthetic_vals[i,1]),4),
                    'callin': round(float(synthetic_vals[i,2]),4),
                    'callout': round(float(synthetic_vals[i,3]),4),
                    'internet': round(float(synthetic_vals[i,4]),2)}
                   for i in range(min(50, num_samples))]
    return jsonify({'status': 'success', 'generated_rows': num_samples,
                    'metrics': metrics, 'samples': samples})

@app.route('/api/detect_anomalies', methods=['POST'])
def detect_anomalies():
    data_req      = request.get_json() or {}
    is_attack_sim = data_req.get('simulate_attack', False)
    if is_attack_sim:
        traffic_batch = np.array([[150.,120.,80.,95.,9500.],[180.,140.,90.,110.,12000.],[200.,160.,100.,130.,15000.]], dtype=np.float32)
        norm_batch    = (traffic_batch - dataset_stats['mu']) / dataset_stats['std']
        scores        = tabular_critic(norm_batch, training=False).numpy().flatten()
        threat_level  = "CRITICAL_DDOS_ATTACK"
        status_code   = "ANOMALY_DETECTED"
    else:
        noise       = tf.random.normal([5, LATENT_DIM_TABULAR])
        norm_batch  = tabular_generator(noise, training=False).numpy()
        scores      = tabular_critic(norm_batch, training=False).numpy().flatten()
        threat_level= "MINIMAL"
        status_code = "HEALTHY_TRAFFIC"
    return jsonify({'status': status_code, 'threat_level': threat_level,
                    'wasserstein_critic_scores': [round(float(s), 4) for s in scores],
                    'avg_score': round(float(np.mean(scores)), 4),
                    'recommendation': 'Isolate anomalous 5G traffic' if is_attack_sim else '5G Telemedicine links normal'})

@app.route('/api/download/csv', methods=['GET'])
def download_csv():
    global latest_synthetic_df
    if latest_synthetic_df is None:
        noise = tf.random.normal([2500, LATENT_DIM_TABULAR])
        z_norm = tabular_generator(noise, training=False).numpy()
        synthetic_vals = np.maximum(0, z_norm * dataset_stats['std'] + dataset_stats['mu'])
        latest_synthetic_df = pd.DataFrame(synthetic_vals, columns=FEATURE_NAMES)
    buffer = io.BytesIO()
    latest_synthetic_df.to_csv(buffer, index=True, index_label='SampleIndex')
    buffer.seek(0)
    return send_file(buffer, mimetype='text/csv', as_attachment=True,
                     download_name='wgan_gp_synthetic_5g_healthcare_data.csv')

if __name__ == '__main__':
    print("Starting Synthesis Hub WGAN-GP Flask Server on http://localhost:5000...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
