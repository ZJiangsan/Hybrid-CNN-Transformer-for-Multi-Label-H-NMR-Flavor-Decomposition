#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 18:08:59 2025

@author: Jiangsan Zhao, jiangsan.zhao@nibio.no, NIBIO
"""




import os, os.path, json, time, random, glob, logging
from datetime import datetime
from math import sqrt
import math
# ---------------------------
# Reproducibility (set first)
# ---------------------------
def seed_everything(seed: int = 5433):
    os.environ["PYTHONHASHSEED"] = str(seed)
    # deterministic cuBLAS workspace (required for torch.use_deterministic_algorithms)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    import numpy as _np
    _np.random.seed(seed)
    import torch as _torch
    _torch.manual_seed(seed)
    _torch.cuda.manual_seed_all(seed)
    _torch.backends.cudnn.benchmark = False
    _torch.backends.cudnn.deterministic = True
    _torch.use_deterministic_algorithms(True)

GLOBAL_SEED = 5678
seed_everything(GLOBAL_SEED)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
%matplotlib inline
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import confusion_matrix

# -----------------------------------------------------------------------------
# Project-specific imports
# !git clone https://github.com/yfWang01/DeepMID
os.chdir("DeepMID")
from DeepMID.readBruker import read_bruker_hs_base

# -----------------------------------------------------------------------------
# Utility: checkpoint saver (temperature removed)
def save_checkpoint_sp(model_path, epoch, LR, name, model, optimizer_gen):
    state = {
        'epoch': epoch,
        'lr': LR,
        'state_dict': model.state_dict(),
        'optimizer': optimizer_gen.state_dict(),
    }
    torch.save(state, os.path.join(model_path, f'Decomp_{name}.pth'))

# -----------------------------------------------------------------------------
# Load reference spectra (13 pure flavors) and 16 real formulations (sorted)
print("Reading Bruker data ...")
plant_flavors = read_bruker_hs_base('data/plant_flavors', False, True, False)
known_formulated_flavors = read_bruker_hs_base('data/known_formulated_flavors', False, True, False)

# Build np arrays
spectra_raw_name = []
for p_f in range(len(plant_flavors)):
    pf_i = (plant_flavors[p_f]['fid']).reshape(1, -1)
    spectra_raw_name.append(plant_flavors[p_f]['name'])
    if p_f == 0:
        spectra_raw = pf_i
    else:
        spectra_raw = np.concatenate((spectra_raw, pf_i), axis=0)

# Known formulations to (16, L)
for kf_f in range(len(known_formulated_flavors)):
    kf_f_i = (known_formulated_flavors[kf_f]['fid']).reshape(1, -1)
    if kf_f == 0:
        kff_np = kf_f_i
    else:
        kff_np = np.concatenate((kff_np, kf_f_i), axis=0)

# P95 normalization to match simulation + training
def _p95_norm_rows(x_np: np.ndarray) -> np.ndarray:
    scale = np.percentile(np.abs(x_np), 95, axis=1, keepdims=True) + 1e-8
    return x_np / scale

# kff_np = _p95_norm_rows(kff_np)

##
# plt.figure(figsize=(12,8))
for s_i in range(13,len(kff_np)):
    plt.figure(figsize=(12,8))

    shit_i = kff_np[s_i,:]
    shit_i.shape
    x0 =np.linspace(1, shit_i.shape[0], shit_i.shape[0], endpoint=True) 
    y00 = shit_i
    plt.plot(x0, y00, color = np.random.rand(3,), linestyle = '-.', label = "F_{}".format(s_i))
    plt.legend()
    plt.show()  
# Load labels for real 16 (13 classes)
true_labels = pd.read_csv('../Formulated_Flavor_Ratios_new.csv', index_col=0).values

# -----------------------------------------------------------------------------
# Simulation of formulations (temperature-free, physically valid dilution)

def simulate_formulations_from_refs(
    refs,                       # np.ndarray (C, L) pure spectra
    n=10000,
    seed=1234,
    plan_seed=5433,
    # mix-size: exactly 2..5 components per formulation
    K_range=(2, 5),
    class_probs=None,           # None => uniform over classes
    # --- NEW: stratified K settings ---
    k_alpha=0.8,                # α for weights ∝ C(C,K)^α; set to None to disable
    # ratio model
    equal_ratio_prob=0.8,       # 80% equal-weight mixtures (match real eval)
    dirichlet_alpha_choices=(0.5, 1.0, 2.0),
    # axis jitter
    shift_max=12,
    warp_prob=0.65, warp_knots=(5, 8), warp_jitter=(2, 6),
    stretch_prob=0.5, stretch_range=(0.985, 1.015),
    # lineshape & phase
    gauss_sigma=(0.06, 0.20),
    phase_deg_std=4.0,
    # baseline & ripple
    baseline_sd=(1e-4, 8e-4, 8e-4),   # a0, a1, a2 stds
    ripple_amp=(0.0, 0.006), ripple_freq=(1.0, 3.0),
    # noise & dilution
    noise_std=(2e-4, 6e-4),
    lf_noise_scale=(0.0, 2e-4),
    diluent_levels=(0.01, 1.0),       # <= constrained to [0,1]
    diluent_probs=None,
    diluent_continuous=True,
    # polarity flip (explicit, flip-invariant labels)
    global_flip_prob=0.5,            # 50% chance to flip whole spectrum
    # label threshold
    final_ratio_threshold=0.01,      # presence if |w*(1-d)| > threshold
    # outputs
    p90_norm=0.8,
    return_clean=False,
    return_dilution=True,
):
    """
    Simulate N MR formulations from reference spectra.

    - Dilution d ∈ [0,1], scale = (1-d) ∈ [0,1] (no accidental sign flip).
    - With probability `global_flip_prob`, multiply the *entire* spectrum by -1
      (after artifacts/noise, before normalization) to mimic acquisition polarity.
    - Labels/masks use absolute effective weights: |w*(1-d)|, so they are
      invariant to any global polarity flip.
    - y is the magnitude-normalized vector over *present* components (sum=1 over positives).
    """
    import math
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    rng_plan = np.random.default_rng(plan_seed)
    R = np.asarray(refs, dtype=np.float32)      # (C, L)
    C, L = R.shape

    # Normalize references per spectrum (safety)
    R = R / (np.max(np.abs(R), axis=1, keepdims=True) + 1e-8)

    # Class prior
    p_cls = None
    if class_probs is not None:
        p_cls = np.asarray(class_probs, dtype=np.float32)
        p_cls = p_cls / (p_cls.sum() + 1e-8)

    # ===== STRATIFIED K ALLOCATION =====
    k_min, k_max = int(K_range[0]), int(K_range[1])
    k_vals = np.arange(k_min, k_max + 1, dtype=int)
    if k_alpha is None:
        K_all = rng_plan.integers(k_min, k_max + 1, size=n)
    else:
        comb_counts = np.array([math.comb(C, k) for k in k_vals], dtype=np.float64)
        wK = comb_counts ** float(k_alpha)
        wK = wK / wK.sum()
        counts = rng_plan.multinomial(n, wK)
        K_all = np.repeat(k_vals, counts)
        rng_plan.shuffle(K_all)

    # ===== helpers =====
    def _warp_1d(x):
        if rng.random() >= warp_prob:
            return x
        nK = int(rng.integers(warp_knots[0], warp_knots[1] + 1))
        jmax = int(rng.integers(warp_jitter[0], warp_jitter[1] + 1))
        base = np.linspace(0, L - 1, nK, dtype=np.float32)
        jitter = rng.integers(-jmax, jmax + 1, size=nK).astype(np.float32)
        jitter[0] = 0.0; jitter[-1] = 0.0
        warped = np.clip(base + jitter, 0, L - 1)
        target = np.linspace(0, L - 1, L, dtype=np.float32)
        source = np.interp(target, warped, base)
        return np.interp(source, np.arange(L, dtype=np.float32), x).astype(np.float32)

    def _stretch_compress(x):
        if rng.random() >= stretch_prob:
            return x
        s = float(rng.uniform(*stretch_range))
        target = np.linspace(-1, 1, L, dtype=np.float32)
        source = np.clip(target / s, -1, 1)
        src_idx = (source + 1) * 0.5 * (L - 1)
        return np.interp(src_idx, np.arange(L, dtype=np.float32), x).astype(np.float32)

    def _lowfreq_noise():
        w = rng.normal(0.0, 1.0, L).astype(np.float32)
        return gaussian_filter1d(w, sigma=128)

    # outputs
    S        = np.empty((n, L), dtype=np.float32)
    y_out    = np.zeros((n, C), dtype=np.float32)
    mask_out = np.zeros((n, C), dtype=np.float32)
    sel_list = []
    dil_out  = np.zeros((n,), dtype=np.float32) if return_dilution else None
    S_clean  = np.empty((n, L), dtype=np.float32) if return_clean else None

    # dilution bounds (clipped to [0,1])
    lvls = np.atleast_1d(diluent_levels).astype(np.float32)
    dmin, dmax = float(np.clip(np.min(lvls), 0.0, 1.0)), float(np.clip(np.max(lvls), 0.0, 1.0))

    # ===== simulation loop =====
    for i in tqdm(range(n), desc=f"Simulating formulations (K∈[{K_range[0]},{K_range[1]}])"):
        # choose K & indices
        K = int(K_all[i])
        idx = rng_plan.choice(C, size=K, replace=False, p=p_cls)
        sel_list.append(idx)

        # weights
        if rng.random() < equal_ratio_prob:
            w = np.full(K, 1.0 / K, dtype=np.float32)
        else:
            alpha_val = float(rng.choice(np.asarray(dirichlet_alpha_choices)))
            w = rng.dirichlet(np.full(K, alpha_val, dtype=np.float32)).astype(np.float32)

        # clean mixture
        x_clean = np.zeros(L, dtype=np.float32)
        for k, c in enumerate(idx): x_clean += w[k] * R[c]
        if return_clean: S_clean[i] = x_clean

        # axis ops
        x = x_clean.copy()
        x = _warp_1d(x)
        x = _stretch_compress(x)
        sft = int(rng.integers(-shift_max, shift_max + 1))
        if sft != 0: x = np.roll(x, sft)

        # dilution scale (0..1)
        if diluent_continuous:
            d = float(rng.uniform(dmin, dmax))
        else:
            p = None if diluent_probs is None else np.asarray(diluent_probs, np.float32)
            if p is not None: p = p / (p.sum() + 1e-8)
            d = float(rng.choice(np.linspace(dmin, dmax, num=4), p=p))
        scale = (1.0 - d)          # ∈ [0,1]
        x *= scale

        # ===== labels/mask from |w*(1-d)| =====
        w_eff_abs = np.abs(w * scale)
        pos = w_eff_abs > float(final_ratio_threshold)
        if not np.any(pos):
            pos[np.argmax(w_eff_abs)] = True

        w_mag = w_eff_abs.copy()
        w_mag[~pos] = 0.0
        s_pos = float(w_mag.sum())
        y_pos = (w_mag / s_pos) if s_pos > 0 else w_mag

        y_vec = np.zeros(C, dtype=np.float32)
        y_vec[idx] = y_pos
        y_out[i] = y_vec
        mask_out[i, idx[pos]] = 1.0

        # lineshape + phase
        sig = float(rng.uniform(*gauss_sigma))
        if sig > 0: x = gaussian_filter1d(x, sigma=sig)
        phi = np.deg2rad(rng.normal(0.0, phase_deg_std))
        if abs(phi) > 1e-6:
            x = (np.cos(phi) * x + np.sin(phi) * np.gradient(x).astype(np.float32))

        # baseline + ripple
        t = np.linspace(-1.0, 1.0, L, dtype=np.float32)
        a0, a1, a2 = baseline_sd
        base = rng.normal(0.0, a0) + rng.normal(0.0, a1) * t + rng.normal(0.0, a2) * (t**2)
        A = float(rng.uniform(*ripple_amp))
        if A > 0:
            f  = float(rng.uniform(*ripple_freq))
            ph = float(rng.uniform(0.0, 2*np.pi))
            base += A * np.sin(2*np.pi * f * t + ph)
        x += base

        # noise (after dilution)
        wn = rng.normal(0.0, float(rng.uniform(*noise_std)), L).astype(np.float32)
        x += wn
        lf = float(rng.uniform(*lf_noise_scale))
        if lf > 0: x += lf * _lowfreq_noise()

        # ----- explicit global polarity flip -----
        if rng.random() < float(global_flip_prob):
            x = -x
            # mask_out[i, idx[pos]] = 0.0


        # robust normalization
        if rng.random() < p90_norm:
            sc = float(np.percentile(np.abs(x), 90)); x = x / (sc + 1e-8)
        else:
            sc = float(np.max(np.abs(x))); x = x / (sc + 1e-12)

        S[i] = x
        if return_dilution: dil_out[i] = d

    out = {"S": S, "y": y_out, "mask": mask_out, "sel": sel_list}
    if return_clean:    out["S_clean"] = S_clean
    if return_dilution: out["dilution"] = dil_out
    return out



# -----------------------------------------------------------------------------
# Model
class NMRDecompHybrid(nn.Module):
    def __init__(self, input_len=32724, num_classes=13, cnn_dim=128, transformer_seq_len=750):
        super().__init__()
        self.seq_len = transformer_seq_len
        self.num_classes = num_classes

        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
        )

        self.norm = nn.LayerNorm(cnn_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_dim, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.classifier_mixture = nn.Sequential(
            nn.Linear(64 * cnn_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, x, references=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        f1 = self.cnn1(x); f2 = self.cnn2(f1); f3 = self.cnn3(f2); f4 = self.cnn4(f3); f5 = self.cnn5(f4)
        x = f5.permute(0, 2, 1)
        x = self.norm(x)
        x = self.transformer(x)
        B, T, C = x.shape
        mix_flat = x.reshape(B, -1)
        mix_logits = self.classifier_mixture(mix_flat)
        return mix_logits


###
class NMRDecompCNNOnly(nn.Module):

    def __init__(self, input_len: int = 32724, num_classes: int = 13, cnn_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes

        self.cnn1 = nn.Sequential(
            nn.Conv1d(1, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(cnn_dim, cnn_dim, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, cnn_dim),
            nn.ReLU(),
        )

        self.norm = nn.LayerNorm(cnn_dim)

        # Classifier unchanged: expects 64 * cnn_dim features
        self.classifier_mixture = nn.Sequential(
            nn.Linear(64 * cnn_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, references=None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        f1 = self.cnn1(x)
        f2 = self.cnn2(f1)
        f3 = self.cnn3(f2)
        f4 = self.cnn4(f3)
        f5 = self.cnn5(f4)

        x = f5.permute(0, 2, 1)  # (B, T, C)
        x = self.norm(x)         # keep the same norm position as the hybrid
        B, T, C = x.shape
        assert T == 64, f"CNN-only expects T==64 to match classifier. Got T={T}."
        mix_flat = x.reshape(B, -1)
        mix_logits = self.classifier_mixture(mix_flat)
        return mix_logits


####
class NMRDecompTransformerOnly(nn.Module):

    def __init__(self, input_len: int = 32724, num_classes: int = 13, cnn_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.cnn_dim = cnn_dim
        self.seq_len = 64  # fixed for strict parity with classifier
        # Calculate patch size and total length so we always get exactly 64 tokens
        self.patch_size = math.ceil(input_len / self.seq_len)
        self.total_len = self.patch_size * self.seq_len

        # Patch embedding: (patch_size) -> (cnn_dim)
        self.patch_embed = nn.Linear(self.patch_size, cnn_dim)

        self.norm = nn.LayerNorm(cnn_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cnn_dim, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.classifier_mixture = nn.Sequential(
            nn.Linear(self.seq_len * cnn_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, references=None) -> torch.Tensor:
        # Expect x as (B, L) or (B, 1, L)
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, L)
        B, L = x.shape
        # Pad or crop to total_len so we can reshape into 64 patches of equal size
        if L < self.total_len:
            x = F.pad(x, (0, self.total_len - L))
        elif L > self.total_len:
            x = x[..., : self.total_len]
        # (B, total_len) -> (B, 64, patch_size)
        x = x.view(B, self.seq_len, self.patch_size)
        x = self.patch_embed(x)  # (B, 64, cnn_dim)

        x = self.norm(x)
        x = self.transformer(x)  # (B, 64, cnn_dim)

        mix_flat = x.reshape(B, -1)
        mix_logits = self.classifier_mixture(mix_flat)
        return mix_logits

# -----------------------------------------------------------------------------
# Training configuration & early stopping (temperature removed)
BATCH_SIZE = 100
LR = 1e-3
END_EPOCH = 3_500_000  # upper bound; early stopping will end sooner
PATIENCE = 40
MIN_DELTA = 1e-4
earlystop_enabled = True
best_val_acc = -float("inf")
epochs_no_improve = 0
best_ckpt_name = "cnnTrans_NMR_val_EStop"

# Deterministic DataLoader generator
torch_gen = torch.Generator()
torch_gen.manual_seed(GLOBAL_SEED)

# -----------------------------------------------------------------------------
# Prepare simulated dataset ONCE for reproducibility
seed0 = GLOBAL_SEED
sim = simulate_formulations_from_refs(spectra_raw.astype(np.float32), n=30000, seed=seed0, return_clean=False, return_dilution=True)

composite_spectra = sim["S"]
weights_list = sim["y"]
masks = sim["mask"]
dilutions = sim["dilution"]

NUM_SAMPLES_n = len(composite_spectra)
print(f"Number of samples: {NUM_SAMPLES_n}")

dataset = TensorDataset(torch.tensor(composite_spectra), torch.tensor(weights_list),
                        torch.tensor(masks), torch.tensor(dilutions))
train_len = int(0.8 * NUM_SAMPLES_n)
val_len = NUM_SAMPLES_n - train_len
train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch_gen)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, persistent_workers=False, generator=torch_gen)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, persistent_workers=False)

# Loss (BCE with class balancing) — clip extreme weights
flavor_freq = masks.mean(axis=0)
pos_weight = 1.0 / (flavor_freq + 1e-6)
pos_weight = np.clip(pos_weight, 0.5, 10.0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

# Model & optimizer
encoder = NMRDecompHybrid().to(device).float()
optimizer_encoder = optim.Adam(encoder.parameters(), lr=LR, weight_decay=1e-4)

# Trackers
train_losses, val_losses = [], []
acc_val_final = -np.inf
loss_val_final = np.inf
threshold_val_final = None
epoch_val = -1
accu_l = 0  # for LR decay block

# -----------------------------------------------------------------------------
# Training loop with early stopping (validation-only model selection)
for epoch in range(END_EPOCH):

    # ---------------- Validation pass (select best on val) ----------------
    encoder.eval()
    improved_this_epoch = False

    # threshold_list = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    threshold_list = [0.7] #[round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]
    mrae_out_val_average = None

    for threshold_v in threshold_list:
        correct, total = 0, 0
        mrae_out_val_0 = 0.0

        with torch.no_grad():
            for inputs, _, labels, dils in val_loader:
                input_val = inputs.squeeze().to(device).float()
                label_val = labels.squeeze().to(device).float()

                out_val = encoder(input_val)  # temperature removed
                loss_val_i = criterion(out_val, label_val)
                mrae_out_val_0 += loss_val_i.item()

                preds = (torch.sigmoid(out_val) > threshold_v).float()
                correct += (preds == label_val).sum().item()
                total += label_val.numel()

        mrae_out_val_average = mrae_out_val_0 / max(len(val_loader), 1)
        val_losses.append(mrae_out_val_average)
        val_acc = correct / max(total, 1)
        print(f"[val] epoch={epoch:05d} acc={val_acc:.4f} thr={threshold_v:.2f}")
        print(f"[val] epoch={epoch:05d} loss={mrae_out_val_average:.4f}")
        # Save by validation improvement (with min-delta)
        if loss_val_final >= mrae_out_val_average:
            loss_val_final = mrae_out_val_average
            best_val_acc = val_acc
            acc_val_final = val_acc
            epoch_val = epoch
            threshold_val_final = threshold_v

            save_checkpoint_sp(os.getcwd(), epoch_val, LR, best_ckpt_name, encoder, optimizer_encoder)
            improved_this_epoch = True
    ##
    
    ########################### 16-real evaluation block (diagnostic only)
    # NOTE: Uses the *saved* best-on-val checkpoint and the *val-selected* threshold.
    #       This is for monitoring; it does not influence training/selection.
    # threshold_val_final = 0.95
    model = NMRDecompHybrid().to(device).float()
    resume_model = os.path.join(os.getcwd(), f'Decomp_{best_ckpt_name}.pth')
    if os.path.isfile(resume_model):
        ckpt_model = torch.load(resume_model, map_location=device, weights_only=True)
        model.load_state_dict(ckpt_model['state_dict'])
        model.eval()
        x_real = torch.tensor(kff_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = torch.sigmoid(model(x_real)).detach().cpu().numpy()
        # print(probs)
        pred_binary = (probs >= float(threshold_val_final)).astype(int)
        true_binary = (true_labels > 0).astype(int)
        correct = (pred_binary == true_binary).sum(axis=1)
        accuracy = float(np.mean(correct / true_binary.shape[1]))
        y_true_flat = true_binary.flatten()
        y_pred_flat = pred_binary.flatten()
        tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        print("*"*60)
        print("🔎 Evaluation on 16 Real Formulated Spectra (diagnostic; val-selected τ)")
        print(f"current Threshold: {threshold_val_final:.2f}")
        print(f"current Accuracy:  {accuracy:.3f}")
        print(f"current Recall:    {tpr:.3f}")
        print(f"current FPR:       {fpr:.3f}")
        print("*"*60)

    # Logging
    print("current epoch =", epoch) 
    print("saved val at loss of", loss_val_final)
    print(f'saved Validation Accuracy: {acc_val_final:.4f}, threshold_save: {threshold_val_final:.2f}')
    print("saved val at epoch of", epoch_val)
    print(f'accu_l : {accu_l:.4f}')
    print('LR :', LR)

    # Early stopping book-keeping
    if earlystop_enabled:
        if improved_this_epoch:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"[early-stop] no val gain this epoch ({epochs_no_improve}/{PATIENCE}, best={best_val_acc:.4f})")
        if epochs_no_improve >= PATIENCE:
            print(f"[early-stop] Stopping at epoch {epoch} (best val acc={best_val_acc:.4f}, thr={threshold_val_final}).")
            break

    # ---------------- Train pass ----------------
    encoder.train()
    # Reuse the same optimizer object; do NOT recreate each epoch (keeps Adam state)
    optimizer_encoder.zero_grad(set_to_none=True)
    mrae_out_train_0 = 0.0
    for inputs, _, labels, _dils in train_loader:
        input_train = inputs.squeeze().to(device).float()
        label_train = labels.squeeze().to(device).float()

        optimizer_encoder.zero_grad(set_to_none=True)
        out_train_v = encoder(input_train)
        loss_train_i = criterion(out_train_v, label_train)
        mrae_out_train_0 += loss_train_i.item()
        loss_train_i.backward()
        optimizer_encoder.step()

    mrae_out_train_average = mrae_out_train_0 / max(len(train_loader), 1)
    train_losses.append(mrae_out_train_average)

    np.save("nmr_decomp_cnnTransformer_train_losses_EStop.npy", np.array(train_losses))
    np.save("nmr_decomp_cnnTransformer_val_losses_EStop.npy",   np.array(val_losses))

    # Progress prints
    c_current = datetime.now()
    print(f"\n[epoch {epoch}] time={c_current} train_loss={mrae_out_train_average:.4f} last_val_loss={mrae_out_val_average:.4f} LR={LR}\n")

    # LR decay block (as in your original logic)
    accu_l += 1
    if accu_l > 10:
        accu_l = 0
        for pg in optimizer_encoder.param_groups:
            pg['lr'] *= 0.9
        LR = optimizer_encoder.param_groups[0]['lr']

# -----------------------------------------------------------------------------
# Save a summary of the selected model (no temperature)
summary = {
    "best_val_acc": float(best_val_acc),
    "epoch_val": int(epoch_val),
    "threshold_val_final": float(threshold_val_final if threshold_val_final is not None else -1),
    "checkpoint": os.path.join(os.getcwd(), f"Decomp_{best_ckpt_name}.pth"),
    "seed": GLOBAL_SEED,
    "batch_size": BATCH_SIZE,
}
with open("early_stop_summary_NMR_cnnTransformer_EStop.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved early-stop summary:", summary)



