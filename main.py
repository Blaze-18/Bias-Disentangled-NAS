"""
=============================================================================
Data-Agnostic Bias-Disentangled Structural Embedding Framework
for Zero-Shot Neural Architecture Search
=============================================================================

Authors  : Student ID 2112576112, Student ID 2110876142
Course   : CSE4102
Supervisor: Dr. Md. Ekramul Hamid
University: University of Rajshahi

Benchmark : NATS-Bench Topology Search Space (TSS)
Dataset   : CIFAR-10 (validation accuracies at 200 epochs)
Search Space: 15,625 unique cell-based architectures

Pipeline Stages
---------------
Stage 1 : Structural proxy extraction (SynFlow, Zen-Score, NASWOT, Param Count)
Stage 2 : Regression-based structural bias disentanglement
Stage 3 : PCA whitening and decorrelated embedding construction
Stage 4 : Non-linear surrogate training with pairwise ranking loss
Stage 5 : Data-agnostic inference and evaluation

Usage (Google Colab)
--------------------
1. Mount Google Drive and place the NATS-Bench file at the path in CONFIG.
2. Install dependencies:
       pip install nats_bench xautodl torch scikit-learn scipy tqdm
3. Run:
       python main.py

All intermediate outputs and the trained model are saved to OUTPUT_DIR.
=============================================================================
"""

import os
import math
import random
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# All user-facing settings are collected here. Change these before running.
# =============================================================================

CONFIG = {
    # -------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------
    # Full path to the NATS-Bench TSS pickle file on Google Drive.
    "NATS_BENCH_PATH": "/content/drive/MyDrive/Thesis NAS/NATS-tss-v1_0-3ffb9.pickle.pbz2",
    # Can use DIR link for local device 
    "NATS_BENCH_DIR_PATH":"YOUR_DIR_PATH"
    # Directory where all outputs (proxy scores, model, results) are saved.
    "OUTPUT_DIR": "/content/drive/MyDrive/Thesis NAS/experiment_outputs",

    # -------------------------------------------------------------------
    # Benchmark settings
    # -------------------------------------------------------------------
    # Dataset to query ground-truth accuracies from.
    # Options: "cifar10-valid", "cifar100", "ImageNet16-120"
    "DATASET": "cifar10-valid",

    # Number of training epochs used in the benchmark for ground-truth.
    # NATS-Bench provides results at hp="12" and hp="200".
    "HP_EPOCHS": "200",

    # -------------------------------------------------------------------
    # Proxy extraction settings
    # -------------------------------------------------------------------
    # Number of architectures to extract proxies for.
    # Set to 15625 for the full search space.
    # Use a smaller number (e.g., 500) for quick testing.
    "NUM_ARCHITECTURES": 15625,

    # Batch size used when computing Zen-Score and NASWOT proxies.
    # A random tensor of this shape is passed through the network.
    "PROXY_BATCH_SIZE": 16,

    # Input image shape assumed for all architectures.
    # CIFAR-10 uses 3 channels, 32x32 resolution.
    "INPUT_CHANNELS": 3,
    "INPUT_RESOLUTION": 32,

    # Number of random Gaussian perturbations used in Zen-Score computation.
    "ZEN_NUM_PERTURBATIONS": 20,

    # -------------------------------------------------------------------
    # Surrogate training settings
    # -------------------------------------------------------------------
    # Number of architectures sampled from the full space to train
    # the surrogate predictor. The thesis uses 800.
    "SURROGATE_TRAIN_SIZE": 800,

    # Fraction of SURROGATE_TRAIN_SIZE used for validation.
    # 0.20 means 80% train, 20% validation.
    "VALIDATION_SPLIT": 0.20,

    # MLP architecture.
    "MLP_HIDDEN_1": 64,
    "MLP_HIDDEN_2": 32,
    "MLP_DROPOUT": 0.10,

    # Training hyperparameters.
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "BATCH_SIZE": 128,
    "NUM_EPOCHS": 100,

    # Weighting between MSE loss and pairwise ranking loss.
    # Final loss = ALPHA * MSE + (1 - ALPHA) * RankingLoss
    "ALPHA": 0.5,

    # -------------------------------------------------------------------
    # Evaluation settings
    # -------------------------------------------------------------------
    # K value used for Top-K selection accuracy.
    "TOP_K": 50,

    # -------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------
    "SEED": 42,
}


# =============================================================================
# SETUP: Logging, seeds, output directory
# =============================================================================

def setup(config):
    """
    Configure logging, set all random seeds for reproducibility,
    and create the output directory if it does not already exist.
    """
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config["OUTPUT_DIR"], "run.log")),
        ],
    )

    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["SEED"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Device: %s", device)
    return device


# =============================================================================
# STAGE 0: Load NATS-Bench API and ground-truth accuracies
# =============================================================================

def load_benchmark(config):
    """
    Load the NATS-Bench TSS API and extract ground-truth validation
    accuracies for all architectures in the search space.

    Returns
    -------
    api : NATSBench
        The loaded benchmark API object.
    ground_truth : np.ndarray, shape (NUM_ARCHITECTURES,)
        Validation accuracy (%) for each architecture.
    """
    from nats_bench import create

    logging.info("Loading NATS-Bench API from: %s", config["NATS_BENCH_PATH"])
    api = create(config["NATS_BENCH_PATH"], "tss", fast_mode=False, verbose=False)
    logging.info("Benchmark loaded. Total architectures: %d", len(api))

    n = config["NUM_ARCHITECTURES"]
    logging.info("Extracting ground-truth accuracies for %d architectures ...", n)

    ground_truth = np.zeros(n, dtype=np.float32)
    for i in tqdm(range(n), desc="Ground truth"):
        info = api.get_more_info(
            i,
            config["DATASET"],
            hp=config["HP_EPOCHS"],
        )
        ground_truth[i] = info["valid-accuracy"]

    logging.info(
        "Accuracy range: %.2f%% to %.2f%%  (mean: %.2f%%)",
        ground_truth.min(),
        ground_truth.max(),
        ground_truth.mean(),
    )

    save_path = os.path.join(config["OUTPUT_DIR"], "ground_truth.npy")
    np.save(save_path, ground_truth)
    logging.info("Ground truth saved to: %s", save_path)

    return api, ground_truth


# =============================================================================
# STAGE 1: Structural proxy extraction
# =============================================================================

def get_model(api, arch_idx, dataset, device):
    """
    Instantiate a NATS-Bench architecture as a PyTorch model.

    Parameters
    ----------
    api : NATSBench
        The loaded benchmark API.
    arch_idx : int
        Index of the architecture in the search space.
    dataset : str
        Dataset string used to retrieve the network configuration.
    device : torch.device
        Device to place the model on.

    Returns
    -------
    model : nn.Module
        The instantiated model with random weights (no training).
    """
    from xautodl.models import get_cell_based_tiny_net

    config = api.get_net_config(arch_idx, dataset)
    model  = get_cell_based_tiny_net(config)
    model  = model.to(device)
    model.eval()
    return model


def compute_param_count(model):
    """
    Count the total number of trainable parameters in the model.

    This is used directly as a structural proxy. It requires no forward
    pass and is O(1) in computation.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    count : float
        Total number of trainable parameters.
    """
    return float(sum(p.numel() for p in model.parameters() if p.requires_grad))


def compute_synflow(model, input_shape, device):
    """
    Compute the SynFlow score for the given architecture.

    SynFlow measures synaptic saliency by computing the sum of the
    element-wise product of each parameter and its gradient with respect
    to a scalar output derived from an all-ones input. This is entirely
    data-agnostic because the input tensor is a constant (all ones),
    not sampled from any dataset.

    The score is defined as:
        SynFlow = sum over all params of |param * grad(R / param)|
    where R is the sum of all output values.

    Reference: Tanaka et al., NeurIPS 2020.

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple
        Shape of the dummy input tensor (batch, channels, height, width).
    device : torch.device

    Returns
    -------
    score : float
        The SynFlow score. Higher is better.
    """
    # All-ones input ensures no data dependency.
    x = torch.ones(input_shape, device=device, requires_grad=False)

    # Linearize all parameters: take absolute values so that sign
    # differences between weights do not cause cancellation.
    original_params = []
    for param in model.parameters():
        original_params.append(param.data.clone())
        param.data = param.data.abs()

    # Enable gradient computation for all parameters.
    for param in model.parameters():
        param.requires_grad_(True)

    # Forward pass.
    model.zero_grad()
    try:
        output = model(x)
        # Handle architectures that return a tuple (logits, aux).
        if isinstance(output, (tuple, list)):
            output = output[0]
        # Scalar: sum all output elements.
        R = output.sum()
        R.backward()
    except Exception:
        # If forward pass fails for any architecture, return 0.
        score = 0.0
        for param, original in zip(model.parameters(), original_params):
            param.data = original
            param.requires_grad_(False)
        return score

    # SynFlow score: sum of |param * grad|.
    score = 0.0
    for param in model.parameters():
        if param.grad is not None:
            score += (param.data * param.grad).abs().sum().item()

    # Restore original parameter values.
    for param, original in zip(model.parameters(), original_params):
        param.data = original
        param.requires_grad_(False)

    model.zero_grad()
    return score


def compute_zen_score(model, input_shape, num_perturbations, device):
    """
    Compute the Zen-Score for the given architecture.

    Zen-Score measures the expressivity of the network by quantifying
    how the output changes in response to small Gaussian perturbations
    applied to a random input. A higher variance in the output implies
    that the network can express a richer set of functions.

    The score is the mean log standard deviation of the output difference
    across multiple perturbation pairs, averaged over all perturbations.

    This is data-agnostic because both the base input and the perturbation
    are sampled from Gaussian noise, not from any real dataset.

    Reference: Lin et al., "Zen-NAS", ICCV 2021.

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple
    num_perturbations : int
        Number of (input, perturbed_input) pairs to average over.
    device : torch.device

    Returns
    -------
    score : float
        The Zen-Score. Higher generally indicates higher expressivity.
    """
    score = 0.0
    with torch.no_grad():
        for _ in range(num_perturbations):
            # Random base input.
            x     = torch.randn(input_shape, device=device)
            # Small Gaussian perturbation.
            delta = torch.randn_like(x) * 0.001

            try:
                out_x = model(x)
                out_p = model(x + delta)

                if isinstance(out_x, (tuple, list)):
                    out_x = out_x[0]
                if isinstance(out_p, (tuple, list)):
                    out_p = out_p[0]

                # Difference in outputs.
                diff = (out_x - out_p).abs()
                # Log standard deviation of the difference.
                std  = diff.std().item()
                if std > 0:
                    score += math.log(std)
            except Exception:
                continue

    return score / num_perturbations if num_perturbations > 0 else 0.0


def compute_naswot(model, input_shape, device):
    """
    Compute the random-input NASWOT score for the given architecture.

    NASWOT (NAS Without Training) measures the diversity of ReLU
    activation patterns across a batch of inputs. For each input in
    the batch, a binary code is produced indicating which ReLU neurons
    are active (output > 0). Architectures that produce highly distinct
    binary codes for different inputs have a higher capacity to separate
    data.

    The score is the log of the absolute determinant of the kernel matrix
    K where K[i,j] = number of shared active neurons between input i and j.
    A higher log-det indicates more diverse activation patterns.

    Here we use random Gaussian inputs instead of real data, maintaining
    the data-agnostic property. This is justified because the structural
    property (activation pattern diversity) is largely independent of
    the specific distribution of inputs.

    Reference: Mellor et al., ICML 2021.

    Parameters
    ----------
    model : nn.Module
    input_shape : tuple
    device : torch.device

    Returns
    -------
    score : float
        Log determinant of the activation kernel matrix. Higher is better.
    """
    activation_codes = []

    # Register forward hooks on every ReLU layer to capture binary codes.
    hooks   = []
    outputs = []

    def hook_fn(module, inp, out):
        # Binary code: 1 where activation is positive, 0 otherwise.
        outputs.append((out > 0).float().view(out.size(0), -1))

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        try:
            x = torch.randn(input_shape, device=device)
            _ = model(x)
        except Exception:
            for h in hooks:
                h.remove()
            return 0.0

    for h in hooks:
        h.remove()

    if not outputs:
        return 0.0

    # Concatenate all activation codes along the feature dimension.
    codes = torch.cat(outputs, dim=1).cpu()  # shape: (batch, total_relu_units)
    batch_size = codes.shape[0]

    # Build the kernel matrix K where K[i,j] = dot(code_i, code_j).
    K = torch.mm(codes, codes.t()).numpy()

    # Log determinant of K.
    sign, log_det = np.linalg.slogdet(K)
    if sign <= 0:
        return 0.0

    return float(log_det)


def extract_all_proxies(api, ground_truth, config, device):
    """
    Extract all four structural proxies for every architecture in the
    search space and save the results to disk.

    This is the most time-consuming stage. On a free-tier GPU (Colab T4),
    expect approximately 2 to 4 hours for 15,625 architectures.

    The function saves a checkpoint every 500 architectures so that
    extraction can be resumed if the runtime is interrupted.

    Parameters
    ----------
    api : NATSBench
    ground_truth : np.ndarray, shape (N,)
    config : dict
    device : torch.device

    Returns
    -------
    proxy_matrix : np.ndarray, shape (N, 4)
        Columns: [param_count, synflow, zen_score, naswot]
    """
    n             = config["NUM_ARCHITECTURES"]
    batch_size    = config["PROXY_BATCH_SIZE"]
    channels      = config["INPUT_CHANNELS"]
    resolution    = config["INPUT_RESOLUTION"]
    zen_perturbs  = config["ZEN_NUM_PERTURBATIONS"]
    input_shape   = (batch_size, channels, resolution, resolution)
    checkpoint_path = os.path.join(config["OUTPUT_DIR"], "proxy_checkpoint.pkl")

    # Check if a previous checkpoint exists and resume from it.
    if os.path.exists(checkpoint_path):
        logging.info("Resuming proxy extraction from checkpoint: %s", checkpoint_path)
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        proxy_matrix = checkpoint["proxy_matrix"]
        start_idx    = checkpoint["completed"]
        logging.info("Resuming from architecture index %d", start_idx)
    else:
        # Columns: param_count, synflow, zen_score, naswot
        proxy_matrix = np.zeros((n, 4), dtype=np.float32)
        start_idx    = 0

    dataset_key = config["DATASET"]

    for i in tqdm(range(start_idx, n), desc="Proxy extraction", initial=start_idx, total=n):
        model = get_model(api, i, dataset_key, device)

        proxy_matrix[i, 0] = compute_param_count(model)
        proxy_matrix[i, 1] = compute_synflow(model, input_shape, device)
        proxy_matrix[i, 2] = compute_zen_score(model, input_shape, zen_perturbs, device)
        proxy_matrix[i, 3] = compute_naswot(model, input_shape, device)

        # Free GPU memory immediately after each architecture.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save a checkpoint every 500 architectures.
        if (i + 1) % 500 == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"proxy_matrix": proxy_matrix, "completed": i + 1}, f)

    # Save final proxy matrix.
    save_path = os.path.join(config["OUTPUT_DIR"], "proxy_matrix.npy")
    np.save(save_path, proxy_matrix)
    logging.info("Proxy extraction complete. Saved to: %s", save_path)

    # Log basic statistics for each proxy.
    proxy_names = ["Param Count", "SynFlow", "Zen-Score", "NASWOT"]
    for j, name in enumerate(proxy_names):
        col = proxy_matrix[:, j]
        logging.info(
            "%s  |  min: %.4f  max: %.4f  mean: %.4f  std: %.4f",
            name, col.min(), col.max(), col.mean(), col.std(),
        )

    return proxy_matrix


# =============================================================================
# STAGE 2: Structural bias disentanglement
# =============================================================================

def disentangle_bias(proxy_matrix, config):
    """
    Remove the linear structural bias introduced by architectural scale
    (parameter count) from each proxy score.

    For each proxy P_i (SynFlow, Zen-Score, NASWOT), a linear regression
    is fitted:
        P_i = alpha * Param_Count + epsilon_i

    The residual epsilon_i is then used as the bias-corrected proxy signal.
    Parameter Count itself is log-transformed and standardized for use as
    an independent structural feature.

    The fitted regression objects are saved so that identical transformations
    can be applied to unseen architectures during inference.

    Parameters
    ----------
    proxy_matrix : np.ndarray, shape (N, 4)
        Raw proxy scores. Columns: [param_count, synflow, zen_score, naswot]
    config : dict

    Returns
    -------
    corrected_matrix : np.ndarray, shape (N, 4)
        Bias-corrected proxy scores.
        Columns: [log_param_count_normalized, synflow_residual,
                  zen_score_residual, naswot_residual]
    bias_models : dict
        Fitted LinearRegression objects keyed by proxy name.
        Needed at inference time.
    param_scaler : StandardScaler
        Fitted scaler for the log-transformed parameter count feature.
    """
    n = proxy_matrix.shape[0]

    param_count = proxy_matrix[:, 0]
    synflow     = proxy_matrix[:, 1]
    zen_score   = proxy_matrix[:, 2]
    naswot      = proxy_matrix[:, 3]

    # Log-transform parameter count to reduce the effect of extreme values
    # and linearize the scale-proxy relationship.
    log_params = np.log(param_count + 1.0)

    # Standardize log-transformed parameter count.
    param_scaler = StandardScaler()
    log_params_scaled = param_scaler.fit_transform(log_params.reshape(-1, 1)).flatten()

    X = log_params_scaled.reshape(-1, 1)

    bias_models  = {}
    proxy_names  = ["synflow", "zen_score", "naswot"]
    proxy_arrays = [synflow, zen_score, naswot]
    residuals    = []

    for name, values in zip(proxy_names, proxy_arrays):
        reg = LinearRegression()
        reg.fit(X, values)
        predicted = reg.predict(X)
        residual  = values - predicted

        bias_models[name] = reg

        r_before, _ = spearmanr(log_params_scaled, values)
        r_after,  _ = spearmanr(log_params_scaled, residual)
        logging.info(
            "Bias removal | %s  |  Spearman with param_count:  before=%.4f  after=%.4f",
            name, r_before, r_after,
        )

        residuals.append(residual)

    # Build the corrected feature matrix.
    # Column 0: normalized log param count (kept as a structural feature)
    # Columns 1-3: bias-corrected residuals for each proxy
    corrected_matrix = np.column_stack(
        [log_params_scaled] + residuals
    ).astype(np.float32)

    # Save artefacts needed for inference.
    save_path = os.path.join(config["OUTPUT_DIR"], "bias_artefacts.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"bias_models": bias_models, "param_scaler": param_scaler}, f)
    logging.info("Bias disentanglement artefacts saved to: %s", save_path)

    return corrected_matrix, bias_models, param_scaler


# =============================================================================
# STAGE 3: PCA whitening and embedding construction
# =============================================================================

def build_embedding(corrected_matrix, config):
    """
    Apply PCA with whitening to the bias-corrected proxy matrix to produce
    a decorrelated, unit-variance architectural embedding.

    Whitening ensures:
        1. All components have zero mean.
        2. All components have unit variance (prevents any single proxy
           from dominating the predictor due to scale differences).
        3. All components are linearly uncorrelated.

    The transformation is:
        Z = W * (P_corrected - mu)
    where W is the whitening matrix derived from the eigendecomposition
    of the covariance matrix of P_corrected.

    Parameters
    ----------
    corrected_matrix : np.ndarray, shape (N, 4)
    config : dict

    Returns
    -------
    embeddings : np.ndarray, shape (N, 4)
        Whitened PCA embeddings (the architectural fingerprints).
    scaler : StandardScaler
        Fitted standard scaler applied before PCA.
    pca : PCA
        Fitted PCA object with whiten=True.
    """
    n_components = corrected_matrix.shape[1]  # Keep all 4 components.

    # Standardize before PCA so that all features have equal initial weight.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(corrected_matrix)

    # PCA with whitening.
    pca = PCA(n_components=n_components, whiten=True, random_state=config["SEED"])
    embeddings = pca.fit_transform(X_scaled).astype(np.float32)

    # Log explained variance per component.
    ev = pca.explained_variance_ratio_
    logging.info("PCA explained variance per component:")
    for k, ratio in enumerate(ev):
        logging.info("  PC%d: %.2f%%  (cumulative: %.2f%%)", k + 1, ratio * 100, ev[:k+1].sum() * 100)

    # Save PCA artefacts for inference.
    save_path = os.path.join(config["OUTPUT_DIR"], "pca_artefacts.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"scaler": scaler, "pca": pca}, f)
    logging.info("PCA artefacts saved to: %s", save_path)

    return embeddings, scaler, pca


# =============================================================================
# STAGE 4: Surrogate MLP with ranking-aware training
# =============================================================================

class SurrogateMLP(nn.Module):
    """
    Shallow Multi-Layer Perceptron used as the non-linear surrogate predictor.

    Architecture:
        Input(4) -> Linear(64) -> ReLU -> Dropout(0.1)
                 -> Linear(32) -> ReLU
                 -> Linear(1)

    The network maps a 4-dimensional whitened PCA embedding to a scalar
    performance score. The output score is used exclusively for ranking
    (higher score = better architecture); it is not a calibrated accuracy
    prediction.

    Parameters
    ----------
    input_dim  : int, default 4
    hidden_1   : int, default 64
    hidden_2   : int, default 32
    dropout_p  : float, default 0.10
    """

    def __init__(self, input_dim=4, hidden_1=64, hidden_2=32, dropout_p=0.10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def pairwise_ranking_loss(y_pred, y_true):
    """
    Compute the pairwise ranking loss.

    For every pair of architectures (i, j), the loss penalizes predictions
    where the predicted ordering contradicts the ground-truth ordering.

    Formally:
        L_rank = mean over all pairs (i,j) of max(0, -(y_i - y_j)(y_hat_i - y_hat_j))

    A pair contributes zero loss when the predicted ranking direction matches
    the ground-truth direction, and a positive loss proportional to the
    magnitude of the disagreement otherwise.

    This loss is aligned with the ultimate evaluation metric (Spearman rank
    correlation) rather than with point-wise accuracy prediction.

    Parameters
    ----------
    y_pred : torch.Tensor, shape (B,)
        Predicted performance scores.
    y_true : torch.Tensor, shape (B,)
        Ground-truth validation accuracies.

    Returns
    -------
    loss : torch.Tensor (scalar)
    """
    # Pairwise differences: diff_true[i,j] = y_true[i] - y_true[j]
    diff_true = y_true.unsqueeze(1) - y_true.unsqueeze(0)  # (B, B)
    diff_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)  # (B, B)

    # Penalize pairs where sign(diff_true) != sign(diff_pred).
    violations = torch.clamp(-(diff_true * diff_pred), min=0.0)

    return violations.mean()


def combined_loss(y_pred, y_true, alpha):
    """
    Compute the combined MSE and pairwise ranking loss.

    Loss = alpha * MSE(y_pred, y_true) + (1 - alpha) * RankingLoss(y_pred, y_true)

    Parameters
    ----------
    y_pred : torch.Tensor, shape (B,)
    y_true : torch.Tensor, shape (B,)
    alpha  : float
        Weighting coefficient. alpha=0.5 weights both losses equally.

    Returns
    -------
    total_loss : torch.Tensor (scalar)
    mse_loss   : torch.Tensor (scalar)
    rank_loss  : torch.Tensor (scalar)
    """
    mse_loss  = nn.functional.mse_loss(y_pred, y_true)
    rank_loss = pairwise_ranking_loss(y_pred, y_true)
    total     = alpha * mse_loss + (1.0 - alpha) * rank_loss
    return total, mse_loss, rank_loss


def prepare_surrogate_data(embeddings, ground_truth, config):
    """
    Sample the surrogate training subset and split it into train and
    validation sets.

    A random subset of SURROGATE_TRAIN_SIZE architectures is drawn from the
    full search space. 80% of this subset is used for training the surrogate
    and 20% for validation. The remaining architectures (not in the subset)
    form the held-out test set used for final evaluation.

    Parameters
    ----------
    embeddings    : np.ndarray, shape (N, 4)
    ground_truth  : np.ndarray, shape (N,)
    config        : dict

    Returns
    -------
    train_loader  : DataLoader
    val_loader    : DataLoader
    test_indices  : np.ndarray
        Indices of architectures in the held-out test set.
    train_indices : np.ndarray
    val_indices   : np.ndarray
    """
    n         = len(ground_truth)
    train_n   = config["SURROGATE_TRAIN_SIZE"]
    val_frac  = config["VALIDATION_SPLIT"]
    batch_sz  = config["BATCH_SIZE"]
    seed      = config["SEED"]

    rng = np.random.RandomState(seed)

    # Draw the surrogate training pool from the full search space.
    all_indices    = np.arange(n)
    pool_indices   = rng.choice(all_indices, size=train_n, replace=False)

    # Split pool into train and validation.
    n_val          = int(train_n * val_frac)
    n_train        = train_n - n_val
    pool_shuffled  = rng.permutation(pool_indices)
    train_indices  = pool_shuffled[:n_train]
    val_indices    = pool_shuffled[n_train:]

    # The test set is everything NOT in the surrogate pool.
    pool_set       = set(pool_indices.tolist())
    test_indices   = np.array([i for i in all_indices if i not in pool_set])

    logging.info(
        "Data split  |  train: %d  val: %d  test: %d",
        len(train_indices), len(val_indices), len(test_indices),
    )

    def make_loader(indices, shuffle):
        X = torch.tensor(embeddings[indices], dtype=torch.float32)
        y = torch.tensor(ground_truth[indices], dtype=torch.float32)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_sz, shuffle=shuffle)

    train_loader = make_loader(train_indices, shuffle=True)
    val_loader   = make_loader(val_indices,   shuffle=False)

    return train_loader, val_loader, test_indices, train_indices, val_indices


def train_surrogate(train_loader, val_loader, config, device):
    """
    Train the SurrogateMLP using the combined MSE and pairwise ranking loss.

    Early stopping is not implemented here in order to train for the full
    100 epochs as specified in the thesis. The best model (lowest validation
    loss) is tracked and saved throughout training.

    Parameters
    ----------
    train_loader : DataLoader
    val_loader   : DataLoader
    config       : dict
    device       : torch.device

    Returns
    -------
    model        : SurrogateMLP
        The best model (lowest validation loss checkpoint).
    history      : dict
        Dictionary with keys "train_loss", "val_loss", "train_mse",
        "train_rank", "val_mse", "val_rank" each mapping to a list
        of per-epoch values.
    """
    model = SurrogateMLP(
        input_dim=4,
        hidden_1=config["MLP_HIDDEN_1"],
        hidden_2=config["MLP_HIDDEN_2"],
        dropout_p=config["MLP_DROPOUT"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=config["WEIGHT_DECAY"],
    )

    alpha      = config["ALPHA"]
    epochs     = config["NUM_EPOCHS"]
    best_val   = float("inf")
    best_state = None

    history = {
        "train_loss": [], "val_loss": [],
        "train_mse":  [], "train_rank": [],
        "val_mse":    [], "val_rank":   [],
    }

    model_path = os.path.join(config["OUTPUT_DIR"], "best_surrogate.pt")

    for epoch in range(1, epochs + 1):

        # ------------------------------------------------------------------
        # Training pass
        # ------------------------------------------------------------------
        model.train()
        train_total = train_mse_sum = train_rank_sum = 0.0
        n_train_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss, mse, rank = combined_loss(y_pred, y_batch, alpha)
            loss.backward()
            optimizer.step()

            train_total    += loss.item()
            train_mse_sum  += mse.item()
            train_rank_sum += rank.item()
            n_train_batches += 1

        avg_train       = train_total    / n_train_batches
        avg_train_mse   = train_mse_sum  / n_train_batches
        avg_train_rank  = train_rank_sum / n_train_batches

        # ------------------------------------------------------------------
        # Validation pass
        # ------------------------------------------------------------------
        model.eval()
        val_total = val_mse_sum = val_rank_sum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred  = model(X_batch)
                loss, mse, rank = combined_loss(y_pred, y_batch, alpha)

                val_total    += loss.item()
                val_mse_sum  += mse.item()
                val_rank_sum += rank.item()
                n_val_batches += 1

        avg_val      = val_total    / n_val_batches
        avg_val_mse  = val_mse_sum  / n_val_batches
        avg_val_rank = val_rank_sum / n_val_batches

        # ------------------------------------------------------------------
        # Record history
        # ------------------------------------------------------------------
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_mse"].append(avg_train_mse)
        history["train_rank"].append(avg_train_rank)
        history["val_mse"].append(avg_val_mse)
        history["val_rank"].append(avg_val_rank)

        # Save the best model checkpoint.
        if avg_val < best_val:
            best_val   = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, model_path)

        if epoch % 10 == 0 or epoch == 1:
            logging.info(
                "Epoch %3d/%d  |  train: %.4f (mse=%.4f, rank=%.4f)  "
                "|  val: %.4f (mse=%.4f, rank=%.4f)",
                epoch, epochs,
                avg_train, avg_train_mse, avg_train_rank,
                avg_val,   avg_val_mse,   avg_val_rank,
            )

    # Load the best checkpoint before returning.
    model.load_state_dict(best_state)
    model.eval()

    logging.info("Best validation loss: %.4f  |  Model saved to: %s", best_val, model_path)

    # Save training history.
    history_path = os.path.join(config["OUTPUT_DIR"], "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    return model, history


# =============================================================================
# STAGE 5: Data-agnostic inference and evaluation
# =============================================================================

def evaluate(model, embeddings, ground_truth, indices, config, device, split_name):
    """
    Evaluate the trained surrogate on a specified subset of architectures.

    Computes:
        - Spearman rank correlation coefficient
        - Kendall-Tau rank correlation coefficient
        - Top-K selection accuracy

    Top-K selection accuracy is defined as the fraction of the true top-K
    architectures (by ground-truth accuracy) that appear in the predicted
    top-K architectures (by surrogate score).

    Parameters
    ----------
    model        : SurrogateMLP (in eval mode)
    embeddings   : np.ndarray, shape (N, 4)
    ground_truth : np.ndarray, shape (N,)
    indices      : np.ndarray
        Indices of the architectures to evaluate.
    config       : dict
    device       : torch.device
    split_name   : str
        Label used in logging (e.g., "validation", "test").

    Returns
    -------
    results : dict
        Keys: "spearman", "kendall", "top_k_accuracy", "predictions"
    """
    K = config["TOP_K"]

    X       = torch.tensor(embeddings[indices], dtype=torch.float32).to(device)
    y_true  = ground_truth[indices]

    model.eval()
    with torch.no_grad():
        y_pred = model(X).cpu().numpy()

    spearman_rho, _ = spearmanr(y_pred, y_true)
    kendall_tau,  _ = kendalltau(y_pred, y_true)

    # Top-K accuracy: how many of the true top-K appear in the predicted top-K.
    true_topk = set(np.argsort(y_true)[-K:].tolist())
    pred_topk = set(np.argsort(y_pred)[-K:].tolist())
    top_k_acc = len(true_topk & pred_topk) / K

    logging.info(
        "[%s]  Spearman rho: %.4f  |  Kendall tau: %.4f  |  Top-%d Acc: %.4f",
        split_name, spearman_rho, kendall_tau, K, top_k_acc,
    )

    return {
        "spearman":      spearman_rho,
        "kendall":       kendall_tau,
        "top_k_accuracy": top_k_acc,
        "predictions":   y_pred,
    }


def evaluate_individual_proxies(proxy_matrix, ground_truth, config):
    """
    Evaluate each individual proxy against ground-truth accuracy as a
    baseline comparison for the surrogate predictor.

    Computes Spearman and Kendall-Tau for all four raw proxies.

    Parameters
    ----------
    proxy_matrix  : np.ndarray, shape (N, 4)
        Raw proxy scores. Columns: [param_count, synflow, zen_score, naswot]
    ground_truth  : np.ndarray, shape (N,)
    config        : dict

    Returns
    -------
    individual_results : dict
        Keyed by proxy name.
    """
    proxy_names = ["Param Count", "SynFlow", "Zen-Score", "NASWOT"]
    results     = {}
    K           = config["TOP_K"]

    logging.info("Individual proxy evaluation (full search space):")
    for j, name in enumerate(proxy_names):
        scores = proxy_matrix[:, j]
        rho, _ = spearmanr(scores, ground_truth)
        tau, _ = kendalltau(scores, ground_truth)

        true_topk = set(np.argsort(ground_truth)[-K:].tolist())
        pred_topk = set(np.argsort(scores)[-K:].tolist())
        top_k_acc = len(true_topk & pred_topk) / K

        logging.info(
            "  %-15s  Spearman: %.4f  Kendall: %.4f  Top-%d Acc: %.4f",
            name, rho, tau, K, top_k_acc,
        )
        results[name] = {"spearman": rho, "kendall": tau, "top_k_accuracy": top_k_acc}

    return results


def rank_full_search_space(model, embeddings, config, device):
    """
    Apply the trained surrogate to rank all architectures in the search space.

    This function performs the data-agnostic inference step described in
    Stage 5 of the thesis. It processes all architectures in batches to
    avoid GPU memory overflow.

    Parameters
    ----------
    model      : SurrogateMLP
    embeddings : np.ndarray, shape (N, 4)
    config     : dict
    device     : torch.device

    Returns
    -------
    ranked_indices : np.ndarray, shape (N,)
        Architecture indices sorted from best (rank 1) to worst (rank N).
    scores : np.ndarray, shape (N,)
        Raw surrogate scores for all architectures.
    """
    n          = len(embeddings)
    batch_sz   = 512  # Internal batch size for inference. Not user-facing.
    all_scores = []

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_sz):
            end    = min(start + batch_sz, n)
            X_b    = torch.tensor(embeddings[start:end], dtype=torch.float32).to(device)
            scores = model(X_b).cpu().numpy()
            all_scores.append(scores)

    scores         = np.concatenate(all_scores)
    ranked_indices = np.argsort(scores)[::-1].copy()  # Descending: best first.

    logging.info("Full search space ranked. Top-5 architecture indices:")
    for rank, idx in enumerate(ranked_indices[:5], start=1):
        logging.info("  Rank %d: architecture %d  (score: %.4f)", rank, idx, scores[idx])

    # Save ranking results.
    save_path = os.path.join(config["OUTPUT_DIR"], "full_ranking.npy")
    np.save(save_path, ranked_indices)
    logging.info("Full ranking saved to: %s", save_path)

    return ranked_indices, scores


# =============================================================================
# INFERENCE PIPELINE: Score a single new architecture (for search use)
# =============================================================================

def score_single_architecture(
    api,
    arch_idx,
    bias_models,
    param_scaler,
    embedding_scaler,
    pca,
    model,
    config,
    device,
):
    """
    Score a single unseen architecture using the fully trained pipeline.

    This is the function to call inside any search algorithm (evolutionary,
    random, Bayesian) to evaluate a candidate architecture. It performs
    the complete Stage 1 through Stage 5 transformation on a single
    architecture and returns a scalar score.

    No real data is required at any point in this function.

    Returns
    -------
    score : float
        Scalar performance score. Higher means predicted better performance.
    """
    dataset_key = config["DATASET"]
    batch_size  = config["PROXY_BATCH_SIZE"]
    channels    = config["INPUT_CHANNELS"]
    resolution  = config["INPUT_RESOLUTION"]
    zen_perturbs = config["ZEN_NUM_PERTURBATIONS"]
    input_shape  = (batch_size, channels, resolution, resolution)

    # Stage 1: Extract raw proxies.
    arch_model = get_model(api, arch_idx, dataset_key, device)

    param_count = compute_param_count(arch_model)
    synflow     = compute_synflow(arch_model, input_shape, device)
    zen_score   = compute_zen_score(arch_model, input_shape, zen_perturbs, device)
    naswot      = compute_naswot(arch_model, input_shape, device)

    del arch_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 2: Bias disentanglement.
    log_param = np.log(param_count + 1.0)
    log_param_scaled = param_scaler.transform([[log_param]])[0, 0]
    X_bias = np.array([[log_param_scaled]])

    synflow_corrected   = synflow   - bias_models["synflow"].predict(X_bias)[0]
    zen_corrected       = zen_score - bias_models["zen_score"].predict(X_bias)[0]
    naswot_corrected    = naswot    - bias_models["naswot"].predict(X_bias)[0]

    corrected = np.array([[log_param_scaled, synflow_corrected, zen_corrected, naswot_corrected]], dtype=np.float32)

    # Stage 3: PCA whitening embedding.
    X_scaled   = embedding_scaler.transform(corrected)
    embedding  = pca.transform(X_scaled).astype(np.float32)

    # Stage 4/5: Surrogate prediction.
    X_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        score = model(X_tensor).item()

    return score


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    device = setup(CONFIG)

    # --------------------------------------------------------------------------
    # Stage 0: Load benchmark and ground truth.
    # --------------------------------------------------------------------------
    api, ground_truth = load_benchmark(CONFIG)

    # --------------------------------------------------------------------------
    # Stage 1: Structural proxy extraction.
    # Check if a previously computed proxy matrix exists to avoid recomputation.
    # --------------------------------------------------------------------------
    proxy_matrix_path = os.path.join(CONFIG["OUTPUT_DIR"], "proxy_matrix.npy")
    if os.path.exists(proxy_matrix_path):
        logging.info("Found existing proxy matrix at: %s  -- skipping extraction.", proxy_matrix_path)
        proxy_matrix = np.load(proxy_matrix_path)
    else:
        proxy_matrix = extract_all_proxies(api, ground_truth, CONFIG, device)

    # --------------------------------------------------------------------------
    # Stage 2: Structural bias disentanglement.
    # --------------------------------------------------------------------------
    corrected_matrix, bias_models, param_scaler = disentangle_bias(proxy_matrix, CONFIG)

    # --------------------------------------------------------------------------
    # Stage 3: PCA whitening and embedding construction.
    # --------------------------------------------------------------------------
    embeddings, embedding_scaler, pca = build_embedding(corrected_matrix, CONFIG)

    # --------------------------------------------------------------------------
    # Stage 4: Surrogate training.
    # --------------------------------------------------------------------------
    train_loader, val_loader, test_indices, train_indices, val_indices = prepare_surrogate_data(
        embeddings, ground_truth, CONFIG
    )

    surrogate_model, history = train_surrogate(train_loader, val_loader, CONFIG, device)

    # --------------------------------------------------------------------------
    # Stage 5: Evaluation.
    # --------------------------------------------------------------------------

    # Evaluate individual proxies as baseline.
    individual_results = evaluate_individual_proxies(proxy_matrix, ground_truth, CONFIG)

    # Evaluate surrogate on the validation set.
    val_results = evaluate(
        surrogate_model, embeddings, ground_truth,
        val_indices, CONFIG, device, "validation"
    )

    # Evaluate surrogate on the held-out test set.
    test_results = evaluate(
        surrogate_model, embeddings, ground_truth,
        test_indices, CONFIG, device, "test"
    )

    # Rank the full search space.
    ranked_indices, all_scores = rank_full_search_space(
        surrogate_model, embeddings, CONFIG, device
    )

    # --------------------------------------------------------------------------
    # Save all results to disk.
    # --------------------------------------------------------------------------
    results_path = os.path.join(CONFIG["OUTPUT_DIR"], "results_summary.pkl")
    with open(results_path, "wb") as f:
        pickle.dump({
            "individual_proxy_results": individual_results,
            "val_results":             val_results,
            "test_results":            test_results,
            "ranked_indices":          ranked_indices,
            "all_scores":              all_scores,
            "history":                 history,
            "config":                  CONFIG,
        }, f)

    logging.info("All results saved to: %s", results_path)
    logging.info("Pipeline complete.")


if __name__ == "__main__":
    main()
