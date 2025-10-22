#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 20:52:14 2025

@author: nibio
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
# %matplotlib inline
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


# visaulize the library (Figure S1) and mixtures (Figure S2)
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


# Suppose your reference spectra list looks like this:
# ref_list = [{'name': 'Vanillin', 'ppm': ppm_array, 'fid': intensity_array}, ...]

import matplotlib.pyplot as plt
import numpy as np

def plot_stacked_spectra(
    spectra, x_off, title, filename,
    peaks_dict=None, offset=1.2,
    lw=1.0, base_font=12
):
    """
    Plot stacked 1H NMR spectra with integer y-axis ticks and larger fonts.
    spectra: list of dicts with keys ['name', 'ppm', 'fid']
    peaks_dict: optional {name: [ppm_peaks]} for annotation
    offset: vertical shift between spectra
    """
    n = len(spectra)
    plt.figure(figsize=(9, n * 0.4 + 3))

    for i, spec in enumerate(spectra):
        x = spec['ppm']
        y = spec['fid'] / np.max(spec['fid'])  # normalize to 1
        plt.plot(x, y + i * offset, color='black', lw=lw)
        plt.text(x[0] - x_off, i * offset+0.6, spec['name'],
                 va='center', ha='right', fontsize=base_font)

        if peaks_dict and spec['name'] in peaks_dict:
            for p in peaks_dict[spec['name']]:
                plt.axvline(p, color='red', lw=0.8, ls='--')

    # Axes settings
    plt.gca().invert_xaxis()  # NMR standard orientation
    plt.xlabel('Chemical Shift (ppm)', fontsize=base_font + 2)
    plt.ylabel('Spectrum Index', fontsize=base_font + 2)
    plt.yticks(np.arange(0, n * offset, offset),
               [str(i) for i in range(n)], fontsize=base_font)
    plt.xticks(fontsize=base_font)
    # plt.title(title, fontsize=base_font + 4, weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# Example usage:
# --- Figure S3a: 13 reference spectra ---
plot_stacked_spectra(plant_flavors, x_off = 4.5, title='Figure S3a. Reference ¹H NMR spectra of 13 flavor components',
                     filename='FigS1_reference.png', peaks_dict=None)

# --- Figure S3b: 16 mixture spectra ---
plot_stacked_spectra(known_formulated_flavors, x_off = 1.5, title='Figure S3b. ¹H NMR spectra of 16 formulated mixtures',
                     filename='FigS2_mixtures.png', peaks_dict=None)

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.BCEWithLogitsLoss()

# Deterministic DataLoader generator
torch_gen = torch.Generator()
torch_gen.manual_seed(GLOBAL_SEED)

# -----------------------------------------------------------------------------
# Prepare simulated dataset ONCE for reproducibility
seed0 = GLOBAL_SEED
sim = simulate_formulations_from_refs(spectra_raw.astype(np.float32), n=30000, seed=seed0, return_clean=False, return_dilution=True)

# Save to save time
# with open("sim_nmr_30000.pkl", "wb") as f:
#     pickle.dump(sim, f)

# # Load later
# with open("sim_nmr_30000.pkl", "rb") as f:
#     sim = pickle.load(f)
    
    
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




# === POST-TRAIN ANALYSIS (fixed τ) ==
# Paste this immediately after the early-stop summary lines
# ================================
import os, json, csv, time, platform, hashlib, warnings
from datetime import datetime
import numpy as np
import torch
from scipy.optimize import nnls
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    jaccard_score, brier_score_loss, _ranking
)

# ---------- Fixed-threshold policy (pre-specified, not tuned) ----------
FIXED_TAU = 0.70
TAU_POLICY = "fixed_a_priori"  # not tuned on val/test
print(f"[policy] threshold policy = {TAU_POLICY}, τ = {FIXED_TAU:.2f} (pre-specified)")

# ------------------ env fingerprint ------------------
def run_fingerprint():
    info = {
        "machine": platform.node(),
        "system": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": (torch.version.cuda if torch.cuda.is_available() else None),
        "cudnn": torch.backends.cudnn.version(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": int(GLOBAL_SEED) if 'GLOBAL_SEED' in globals() else None,
        "timestamp": str(datetime.now()),
        "script_sha1": None,
    }
    try:
        with open(__file__, "rb") as fh:
            info["script_sha1"] = hashlib.sha1(fh.read()).hexdigest()
    except Exception:
        pass
    print("RUN META:", json.dumps(info, indent=2))
    return info

run_meta = run_fingerprint()

# ------------------ helpers ------------------
def ece_score(y_true, y_prob, n_bins=15):
    y_true_flat = y_true.ravel().astype(np.float32)
    y_prob_flat = y_prob.ravel().astype(np.float32)
    bins = np.linspace(0,1,n_bins+1)
    ece=0.0
    for i in range(n_bins):
        m = (y_prob_flat>=bins[i]) & (y_prob_flat<bins[i+1])
        if m.any():
            conf = float(y_prob_flat[m].mean())
            acc  = float(y_true_flat[m].mean())
            ece += m.mean() * abs(acc - conf)
    return float(ece)

def pick_per_class_thresholds(y_true_val, y_prob_val, grid=np.linspace(0.2,0.9,71)):
    # Exploratory/SI only; NOT used for main metrics
    C = y_true_val.shape[1]
    thr = np.zeros(C, dtype=np.float32)
    for c in range(C):
        best_f1, best_t = -1.0, 0.5
        yt = y_true_val[:, c]; yp = y_prob_val[:, c]
        for t in grid:
            f1 = f1_score(yt, (yp>=t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = best_t
    return thr

def multilabel_metrics(y_true, y_prob, thr=0.7, per_class_thr=None):
    C = y_true.shape[1]
    thr_vec = np.full(C, thr, dtype=np.float32) if per_class_thr is None else np.asarray(per_class_thr, dtype=np.float32)
    y_pred = (y_prob >= thr_vec).astype(int)

    out = {}
    out["subset_acc"]   = float(np.mean((y_pred == y_true).all(axis=1)))
    out["elem_acc"]     = float(np.mean(y_pred == y_true))               # element-wise accuracy
    out["sample_f1"]    = f1_score(y_true, y_pred, average="samples", zero_division=0)
    out["macro_f1"]     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    out["micro_f1"]     = f1_score(y_true, y_pred, average="micro", zero_division=0)
    out["macro_jacc"]   = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
    out["micro_jacc"]   = jaccard_score(y_true, y_pred, average="micro", zero_division=0)
    out["brier"]        = float(np.mean([brier_score_loss(y_true[:,c], y_prob[:,c]) for c in range(C)]))
    out["ece"]          = ece_score(y_true, y_prob, n_bins=15)
    return out, y_pred

def tau_sweep(y_true, y_prob, taus=np.linspace(0.3, 0.9, 25)):
    rows=[]
    for t in taus:
        m,_ = multilabel_metrics(y_true, y_prob, thr=float(t))
        rows.append({"tau": float(t), **m})
    return rows

def bootstrap_metric(y_true, y_prob, metric_name="elem_acc", n=2000, seed=12345, thr=0.7, per_class_thr=None):
    rng = np.random.default_rng(seed)
    N = y_true.shape[0]
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        m,_ = multilabel_metrics(y_true[idx], y_prob[idx], thr=thr, per_class_thr=per_class_thr)
        vals.append(m[metric_name])
    vals = np.array(vals, dtype=np.float32)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(vals.mean()), (float(lo), float(hi))

def mcnemar_multilabel_exactmatch(y_true, y_pred_A, y_pred_B):
    a = (y_pred_A == y_true).all(axis=1)
    b = (y_pred_B == y_true).all(axis=1)
    b01 = int(np.sum((~a) & b))  # A wrong, B right
    b10 = int(np.sum(a & (~b)))  # A right, B wrong
    n = b01 + b10
    if n == 0:
        return 1.0, b01, b10
    from math import comb
    kmax = min(b01, b10)
    p = sum(comb(n, k) for k in range(0, kmax+1)) * (0.5**n) * 2.0
    return float(min(1.0, p)), b01, b10

def safe_macro_auc(y_true, y_prob):
    # average only over classes that have BOTH positives and negatives
    C = y_true.shape[1]
    vals, valid = [], []
    for c in range(C):
        yt = y_true[:, c]
        if yt.min() == yt.max():  # all 0s or all 1s -> ROC undefined
            continue
        vals.append(roc_auc_score(yt, y_prob[:, c]))
        valid.append(c)
    if len(vals) == 0:
        return float("nan"), []
    return float(np.mean(vals)), valid

def safe_macro_auprc(y_true, y_prob):
    # average only over classes that have at least one positive
    C = y_true.shape[1]
    vals, valid = [], []
    for c in range(C):
        yt = y_true[:, c]
        if yt.sum() == 0:
            continue
        vals.append(average_precision_score(yt, y_prob[:, c]))
        valid.append(c)
    if len(vals) == 0:
        return float("nan"), []
    return float(np.mean(vals)), valid

def json_safe(obj):
    """Convert numpy types/arrays to plain Python so json.dump works."""
    import numpy as _np
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, (_np.floating, _np.integer, _np.bool_)):
        return obj.item()
    return obj

def write_csv(path, header, rows):
    newfile = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(header)
        for r in rows:
            w.writerow(r)

# ------------------ load best checkpoint ------------------
best_path = os.path.join(os.getcwd(), "unsupervised_Decomp_learning_cnnTrans_NMR_val_EStop.pth")
assert os.path.isfile(best_path), f"Missing checkpoint: {best_path}"
model = NMRDecompHybrid().to(device).float()
ckpt = torch.load(best_path, map_location=device)
state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

# ------------------ VAL probs/labels (for calibration & exploratory τc) ------------------
y_true_val_list, y_prob_val_list = [], []
with torch.no_grad():
    for inputs, _, labels, _dils in val_loader:
        x = inputs.squeeze().to(device).float()
        y = labels.squeeze().cpu().numpy().astype(int)
        p = torch.sigmoid(model(x)).detach().cpu().numpy().astype(np.float32)
        y_true_val_list.append(y)
        y_prob_val_list.append(p)
y_true_val = np.concatenate(y_true_val_list, axis=0)
y_prob_val = np.concatenate(y_prob_val_list, axis=0)

# Per-class thresholds from VAL (exploratory/SI only)
thr_per_class = pick_per_class_thresholds(y_true_val, y_prob_val)

# VAL metrics (reported for completeness; τ fixed a priori)
val_metrics_fixed,_ = multilabel_metrics(y_true_val, y_prob_val, thr=FIXED_TAU, per_class_thr=None)

# Add ROC/PR (safe macro) + micro where valid
macro_auroc_val, valid_roc_val = safe_macro_auc(y_true_val, y_prob_val)
macro_auprc_val, valid_pr_val  = safe_macro_auprc(y_true_val, y_prob_val)
try:
    micro_auroc_val = float(roc_auc_score(y_true_val, y_prob_val, average="micro"))
except Exception:
    micro_auroc_val = float("nan")
try:
    micro_auprc_val = float(average_precision_score(y_true_val, y_prob_val, average="micro"))
except Exception:
    micro_auprc_val = float("nan")

# ------------------ REAL (16 mixtures) probs/labels ------------------
x_real = torch.tensor(kff_np, dtype=torch.float32, device=device)
with torch.no_grad():
    probs_real = torch.sigmoid(model(x_real)).detach().cpu().numpy().astype(np.float32)
true_binary_real = (true_labels > 0).astype(int)

# Metrics on REAL @ fixed τ (headline)
real_metrics_fixed, pred_bin_fixed = multilabel_metrics(true_binary_real, probs_real, thr=FIXED_TAU, per_class_thr=None)

# Exploratory: metrics with per-class τ (SI only)
real_metrics_pclass, pred_bin_pclass = multilabel_metrics(true_binary_real, probs_real, thr=FIXED_TAU, per_class_thr=thr_per_class)

# Add ROC/PR (safe macro) + micro where valid
macro_auroc_real, valid_roc_real = safe_macro_auc(true_binary_real, probs_real)
macro_auprc_real, valid_pr_real  = safe_macro_auprc(true_binary_real, probs_real)
try:
    micro_auroc_real = float(roc_auc_score(true_binary_real, probs_real, average="micro"))
except Exception:
    micro_auroc_real = float("nan")
try:
    micro_auprc_real = float(average_precision_score(true_binary_real, probs_real, average="micro"))
except Exception:
    micro_auprc_real = float("nan")

# τ-sweep table (REAL) — exploratory, for SI
tau_rows_real = tau_sweep(true_binary_real, probs_real, taus=np.linspace(0.3,0.9,25))
# Save τ-sweep as CSV (no pandas)
ts_header = ["tau","subset_acc","elem_acc","sample_f1","macro_f1","micro_f1","macro_jacc","micro_jacc","brier","ece"]
ts_rows = [[r["tau"], r["subset_acc"], r["elem_acc"], r["sample_f1"], r["macro_f1"], r["micro_f1"],
            r["macro_jacc"], r["micro_jacc"], r["brier"], r["ece"]] for r in tau_rows_real]
write_csv("tau_sweep_real.csv", ts_header, ts_rows)
print("Saved: tau_sweep_real.csv")

# Bootstrapped 95% CI on REAL (elem_acc and micro_f1), fixed τ (headline) and per-class τ (SI)
boot_elem_mean_fx, boot_elem_ci_fx = bootstrap_metric(true_binary_real, probs_real, "elem_acc", n=2000, thr=FIXED_TAU)
boot_mif1_mean_fx, boot_mif1_ci_fx = bootstrap_metric(true_binary_real, probs_real, "micro_f1", n=2000, thr=FIXED_TAU)
boot_elem_mean_pc, boot_elem_ci_pc = bootstrap_metric(true_binary_real, probs_real, "elem_acc", n=2000, per_class_thr=thr_per_class)
boot_mif1_mean_pc, boot_mif1_ci_pc = bootstrap_metric(true_binary_real, probs_real, "micro_f1", n=2000, per_class_thr=thr_per_class)

# Which classes are absent on REAL?
absent_pos = np.where(true_binary_real.sum(axis=0) == 0)[0]
absent_neg = np.where((1 - true_binary_real).sum(axis=0) == 0)[0]

# ------------------ optional significance vs. baselines (if *.npy provided) ------------------
sig_tests = {}
try:
    if os.path.isfile("probs_real_cnnOnly.npy"):
        probs_cnnOnly = np.load("probs_real_cnnOnly.npy")
        _, pred_cnnOnly = multilabel_metrics(true_binary_real, probs_cnnOnly, thr=FIXED_TAU)
        p, b01, b10 = mcnemar_multilabel_exactmatch(true_binary_real, pred_bin_fixed, pred_cnnOnly)
        sig_tests["cnnTransformer_vs_cnnOnly_tau_fixed"] = {"p_value": p, "A_wrong_B_right": b01, "A_right_B_wrong": b10}
    if os.path.isfile("probs_real_transformerOnly.npy"):
        probs_transOnly = np.load("probs_real_transformerOnly.npy")
        _, pred_transOnly = multilabel_metrics(true_binary_real, probs_transOnly, thr=FIXED_TAU)
        p, b01, b10 = mcnemar_multilabel_exactmatch(true_binary_real, pred_bin_fixed, pred_transOnly)
        sig_tests["cnnTransformer_vs_transformerOnly_tau_fixed"] = {"p_value": p, "A_wrong_B_right": b01, "A_right_B_wrong": b10}
except Exception as e:
    print("Significance check skipped due to error:", e)

# ------------------ NNLS linearity & reconstruction consistency ------------------

R = spectra_raw.astype(np.float32)  # (C, L)
R = R / (np.max(np.abs(R), axis=1, keepdims=True) + 1e-8)
Xn = kff_np.astype(np.float32) #_p95_norm_rows()

R2_full, R2_pred_fixed, R2_pred_pclass = [], [], []
for i in range(Xn.shape[0]):
    y = Xn[i]
    # unconstrained NNLS on all 13 refs
    w_all, _ = nnls(R.T, y)
    yhat_all = w_all @ R
    ss_res = np.sum((y - yhat_all)**2)
    ss_tot = np.sum((y - y.mean())**2) + 1e-8
    R2_full.append(1.0 - ss_res/ss_tot)

    # restricted to predicted labels (fixed τ)
    sel_fx = np.where(pred_bin_fixed[i] == 1)[0]
    if len(sel_fx) > 0:
        w_fx, _ = nnls(R[sel_fx].T, y)
        yhat_fx = w_fx @ R[sel_fx]
        ss_res_fx = np.sum((y - yhat_fx)**2)
        R2_pred_fixed.append(1.0 - ss_res_fx/ss_tot)
    else:
        R2_pred_fixed.append(0.0)

    # restricted to predicted labels (per-class τ, exploratory)
    sel_pc = np.where(pred_bin_pclass[i] == 1)[0]
    if len(sel_pc) > 0:
        w_pc, _ = nnls(R[sel_pc].T, y)
        yhat_pc = w_pc @ R[sel_pc]
        ss_res_pc = np.sum((y - yhat_pc)**2)
        R2_pred_pclass.append(1.0 - ss_res_pc/ss_tot)
    else:
        R2_pred_pclass.append(0.0)

nnls_summary = {
    "R2_full_mean": float(np.mean(R2_full)), "R2_full_sd": float(np.std(R2_full)),
    "R2_pred_fixed_mean": float(np.mean(R2_pred_fixed)), "R2_pred_fixed_sd": float(np.std(R2_pred_fixed)),
    "R2_pred_pclass_mean": float(np.mean(R2_pred_pclass)), "R2_pred_pclass_sd": float(np.std(R2_pred_pclass)),
    "R2_full_per_sample": list(np.round(R2_full, 4)),
    "R2_pred_fixed_per_sample": list(np.round(R2_pred_fixed, 4)),
    "R2_pred_pclass_per_sample": list(np.round(R2_pred_pclass, 4)),
}
# Save per-sample NNLS as CSV (no pandas)
write_csv(
    "nnls_per_sample.csv",
    ["R2_full","R2_pred_fixed","R2_pred_pclass"],
    list(zip(nnls_summary["R2_full_per_sample"], nnls_summary["R2_pred_fixed_per_sample"], nnls_summary["R2_pred_pclass_per_sample"]))
)
print("Saved: nnls_per_sample.csv")

# ------------------ efficiency (params + latency) ------------------
n_params = sum(p.numel() for p in model.parameters())
dummy = torch.randn(1, 1, 32724, device=device)
# warmup
with torch.no_grad():
    for _ in range(10): _ = model(dummy)
if device=="cuda":
    torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(50): _ = model(dummy)
if device=="cuda":
    torch.cuda.synchronize()
avg_latency_ms = 1000.0 * (time.time() - t0) / 50.0

efficiency = {
    "params_M": float(n_params/1e6),
    "avg_latency_ms_single_spectrum": float(avg_latency_ms),
    "device": device,
}








# ------------------ aggregate report (headline uses FIXED_TAU) ------------------
report = {
    "run_meta": run_meta,
    "thresholds": {
        "policy": TAU_POLICY,
        "global_tau": float(FIXED_TAU),
        "per_class_tau": list(np.round(thr_per_class, 3)),  # exploratory (SI)
    },
    "validation": {
        "metrics_fixed_tau": val_metrics_fixed,
        "macro_auroc": macro_auroc_val,
        "macro_auprc": macro_auprc_val,
        "micro_auroc": micro_auroc_val,
        "micro_auprc": micro_auprc_val,
        "macro_valid_classes": {"auroc": valid_roc_val, "auprc": valid_pr_val},
    },
    "real16": {
        "metrics_fixed_tau": real_metrics_fixed,           # HEADLINE
        "metrics_per_class_tau": real_metrics_pclass,      # exploratory/SI
        "macro_auroc": macro_auroc_real,
        "macro_auprc": macro_auprc_real,
        "micro_auroc": micro_auroc_real,
        "micro_auprc": micro_auprc_real,
        "macro_valid_classes": {"auroc": valid_roc_real, "auprc": valid_pr_real},
        "absent_classes": {"no_positives": absent_pos.tolist(), "no_negatives": absent_neg.tolist()},
        "tau_sweep": tau_rows_real,  # exploratory/SI
        "bootstrap": {
            "elem_acc_fixed": {"mean": boot_elem_mean_fx, "ci95": boot_elem_ci_fx},
            "micro_f1_fixed": {"mean": boot_mif1_mean_fx, "ci95": boot_mif1_ci_fx},
            "elem_acc_pclass": {"mean": boot_elem_mean_pc, "ci95": boot_elem_ci_pc},
            "micro_f1_pclass": {"mean": boot_mif1_mean_pc, "ci95": boot_mif1_ci_pc},
        },
        "significance_optional": sig_tests,
        "nnls": nnls_summary,
    },
    "efficiency": efficiency,
}

# JSON-safe dump
with open("nmr_post_train_metrics_report.json", "w") as f:
    json.dump(json_safe(report), f, indent=2)
print("Saved: nmr_post_train_metrics_report.json")

# ---------- robust CSV append (schema-aware) ----------
def append_row_schema_aware(primary_path, header, row_vals):
    import os
    # If file missing, write with header
    if not os.path.isfile(primary_path):
        with open(primary_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerow(row_vals)
        print(f"Appended row to {primary_path}")
        return
    # read first line to check schema
    with open(primary_path, "r", newline="") as f:
        first = f.readline().strip()
    existing_cols = [c.strip() for c in first.split(",")]
    if existing_cols == header:
        with open(primary_path, "a", newline="") as f:
            csv.writer(f).writerow(row_vals)
        print(f"Appended row to {primary_path}")
    else:
        alt = os.path.splitext(primary_path)[0] + "_v2.csv"
        if not os.path.isfile(alt):
            with open(alt, "w", newline="") as f:
                w = csv.writer(f); w.writerow(header); w.writerow(row_vals)
        else:
            with open(alt, "a", newline="") as f:
                csv.writer(f).writerow(row_vals)
        print(f"Header mismatch in {primary_path}. Wrote to {alt} instead.")

csv_header = [
    "timestamp","machine","seed","device",
    "tau_policy","tau_used",
    "real_elem_acc_tau_fixed","real_micro_f1_tau_fixed","real_subset_acc_tau_fixed",
    "val_elem_acc_tau_fixed","val_micro_f1_tau_fixed",
    "params_M","latency_ms"
]
csv_row = [
    str(report["run_meta"]["timestamp"]),
    str(report["run_meta"]["machine"]),
    int(report["run_meta"]["seed"]) if report["run_meta"]["seed"] is not None else "",
    str(report["run_meta"]["device"]),
    report["thresholds"]["policy"], float(report["thresholds"]["global_tau"]),
    float(report["real16"]["metrics_fixed_tau"]["elem_acc"]),
    float(report["real16"]["metrics_fixed_tau"]["micro_f1"]),
    float(report["real16"]["metrics_fixed_tau"]["subset_acc"]),
    float(report["validation"]["metrics_fixed_tau"]["elem_acc"]),
    float(report["validation"]["metrics_fixed_tau"]["micro_f1"]),
    float(report["efficiency"]["params_M"]),
    float(report["efficiency"]["avg_latency_ms_single_spectrum"]),
]
append_row_schema_aware("robustness_runs.csv", csv_header, csv_row)
# ================================
# === END POST-TRAIN ANALYSIS ====
# ================================





##################################################################
#################################################################

import csv, os

OLD = ["timestamp","machine","seed","device",
       "real_elem_acc_tau0.70","real_micro_f1_tau0.70","real_subset_acc_tau0.70",
       "val_elem_acc_tau0.70","val_micro_f1_tau0.70","params_M","latency_ms"]

NEW = ["timestamp","machine","seed","device",
       "tau_policy","tau_used",
       "real_elem_acc_tau_fixed","real_micro_f1_tau_fixed","real_subset_acc_tau_fixed",
       "val_elem_acc_tau_fixed","val_micro_f1_tau_fixed","params_M","latency_ms"]

in_path  = "robustness_runs.csv"
out_path = "robustness_runs_v2.csv"

assert os.path.isfile(in_path), f"Missing {in_path}"

with open(in_path, "r", newline="") as f:
    header = next(csv.reader(f))
if header == NEW:
    open(out_path, "w").write(open(in_path).read())
    print(f"Wrote {out_path} (already new schema).")
elif header == OLD:
    with open(in_path, "r", newline="") as f, open(out_path, "w", newline="") as g:
        r, w = csv.reader(f), csv.writer(g)
        _ = next(r)  # skip old header
        w.writerow(NEW)
        for row in r:
            (ts, mach, seed, dev,
             real_acc, real_microf1, real_subset,
             val_acc, val_microf1, params_m, lat_ms) = row
            new_row = [ts, mach, seed, dev,
                       "fixed_a_priori", "0.70",
                       real_acc, real_microf1, real_subset,
                       val_acc, val_microf1, params_m, lat_ms]
            w.writerow(new_row)
    print(f"Upgraded → {out_path}")
else:
    raise SystemExit(f"Unexpected header in {in_path}:\n{header}\nExpected either:\n{OLD}\n or \n{NEW}")



##################################################################
#################################################################
#################################################################

import os, json, csv
JSON_PATH = "nmr_post_train_metrics_report.json"
assert os.path.isfile(JSON_PATH), f"Can't find {JSON_PATH}"
R = json.load(open(JSON_PATH))

def rget(d,*keys,default=None):
    for k in keys:
        d = d.get(k, {})
    return d if d else default

# 3.1 write per-class thresholds (csv + optional names if available later)
thr = R["thresholds"]["per_class_tau"]
with open("per_class_thresholds.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["class_index","tau_star"])
    for i,t in enumerate(thr): w.writerow([i, t])
print("Saved: per_class_thresholds.csv")

# 3.2 τ-sweep already saved by analysis block (tau_sweep_real.csv)

# 3.3 NNLS per-sample already saved by analysis block (nnls_per_sample.csv)

# 3.4 Efficiency table (markdown, no extra deps)
eff = R["efficiency"]; rm = R["real16"]["metrics_fixed_tau"]; vm = R["validation"]["metrics_fixed_tau"]
cols = ["params_M","latency_ms_gpu","real_elem_acc_tau_fixed","real_micro_f1_tau_fixed",
        "real_subset_acc_tau_fixed","val_elem_acc_tau_fixed","val_micro_f1_tau_fixed"]
vals = [
    eff["params_M"], eff["avg_latency_ms_single_spectrum"],
    rm["elem_acc"], rm["micro_f1"], rm["subset_acc"],
    vm["elem_acc"], vm["micro_f1"]
]
hdr = "| " + " | ".join(cols) + " |\n" + "|" + " | ".join(["---"]*len(cols)) + " |\n"
row = "| " + " | ".join(f"{float(v):.6f}" if isinstance(v,(int,float)) else str(v) for v in vals) + " |\n"
open("efficiency_table.md","w").write(hdr+row)
print("Saved: efficiency_table.md")

# 3.5 Results summary (markdown)
ci = R["real16"]["bootstrap"]
summary = f"""\
**Fixed threshold:** τ={R['thresholds']['global_tau']} (policy: {R['thresholds']['policy']}).

**Real (16 mixtures):**
- elem-acc = {rm['elem_acc']:.6f}  (95% CI {tuple(ci['elem_acc_fixed']['ci95'])})
- micro-F1 = {rm['micro_f1']:.6f}  (95% CI {tuple(ci['micro_f1_fixed']['ci95'])})
- subset-acc = {rm['subset_acc']:.6f}

**Validation (sim):**
- elem-acc = {vm['elem_acc']:.6f}
- micro-F1 = {vm['micro_f1']:.6f}

**Calibration (ECE / Brier):**
- VAL fixed-τ: ECE {R['validation']['metrics_fixed_tau']['ece']:.6f}, Brier {R['validation']['metrics_fixed_tau']['brier']:.6f}
- REAL fixed-τ: ECE {rm['ece']:.6f}, Brier {rm['brier']:.6f}

**NNLS (linearity & reconstruction):**
- R²_full mean±sd = {R['real16']['nnls']['R2_full_mean']:.4f} ± {R['real16']['nnls']['R2_full_sd']:.4f}
- R²_pred (fixed-τ) mean±sd = {R['real16']['nnls']['R2_pred_fixed_mean']:.4f} ± {R['real16']['nnls']['R2_pred_fixed_sd']:.4f}
"""
open("results_summary.md","w").write(summary)
print("Saved: results_summary.md")





##  supplementary Figure S6 and S7
# =========================
# =========================
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# -------- config --------
TAU = 0.70
SAVE_DIR = "si_figs"
CLASS_NAMES = None
PPM_GRID = None
os.makedirs(SAVE_DIR, exist_ok=True)
#
# Loss (BCE with class balancing) — clip extreme weights
flavor_freq = masks.mean(axis=0)
pos_weight = 1.0 / (flavor_freq + 1e-6)
pos_weight = np.clip(pos_weight, 0.5, 10.0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
# ------------------ load best checkpoint ------------------
best_path = os.path.join(os.getcwd(), "unsupervised_Decomp_learning_cnnTrans_NMR_val_EStop.pth")
assert os.path.isfile(best_path), f"Missing checkpoint: {best_path}"
model = NMRDecompHybrid().to(device).float()
ckpt = torch.load(best_path, map_location=device)
state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
model.load_state_dict(state_dict)
model.eval()

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mtick

# ---------------------- plot_case (unchanged style; ppm + class integer ticks) ----------------------
def plot_case(x, y, p, tau=0.70, class_names=None, ppm=None, title="", fname=None,
              base_fs=18, title_fs=20, label_fs=18, tick_fs=16):
    L = x.shape[-1]
    ppm = known_formulated_flavors[1]['ppm']
    if ppm is None:
        ppm = np.arange(L)

    with mpl.rc_context({
        "font.size": base_fs,
        "axes.titlesize": title_fs,
        "axes.labelsize": label_fs,
        "xtick.labelsize": tick_fs,
        "ytick.labelsize": tick_fs,
        "legend.fontsize": tick_fs,
    }):
        fig, axs = plt.subplots(2, 1, figsize=(6.2, 5.8),
                                gridspec_kw={"height_ratios":[2.1, 1.0]})

        # spectrum
        axs[0].plot(ppm, x)
        try: axs[0].invert_xaxis()
        except: pass
        # >>> integer ticks on ppm
        axs[0].xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        axs[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))
        # <<<
        axs[0].set_ylabel("intensity")
        axs[0].set_xlabel("ppm")
        axs[0].set_title(title)

        # probabilities
        idx = np.arange(len(p))
        axs[1].bar(idx, p)
        axs[1].axhline(tau, linestyle="--")
        axs[1].set_ylim(0, 1)
        axs[1].set_ylabel("P(class)")
        axs[1].set_xlabel("class")
        # Force integer ticks at every class index
        axs[1].set_xticks(idx)
        axs[1].xaxis.set_major_formatter(mtick.FormatStrFormatter('%d'))

        for ax in axs:
            ax.tick_params(length=4, width=1)

        fig.tight_layout()
        if fname:
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            return fig

# ---------------------- helpers (unchanged) ----------------------
def pick_worst_fn(Y, P, tau=0.70):
    mask = (Y == 1) & (P < tau)
    if not mask.any(): return None
    i, c = np.unravel_index(np.argmin(np.where(mask, P, 1.0)), P.shape)
    return i, c, "Worst false negative"

def pick_worst_fp(Y, P, tau=0.70):
    mask = (Y == 0) & (P >= tau)
    if not mask.any(): return None
    i, c = np.unravel_index(np.argmax(np.where(mask, P, -1.0)), P.shape)
    return i, c, "Worst false positive"

def pick_closest_to_tau(Y, P, tau=0.70):
    i, c = np.unravel_index(np.argmin(np.abs(P - tau)), P.shape)
    return i, c, "Closest-to-τ ambiguous"

def summarize_binary(y_true_bin, y_pred_bin):
    y_true_flat = y_true_bin.flatten()
    y_pred_flat = y_pred_bin.flatten()
    tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat).ravel()
    acc = (y_pred_bin == y_true_bin).mean()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(acc), float(tpr), float(fpr)

# ------------------ VAL evaluation (unchanged) ------------------
val_losses = []
val_X, val_Y, val_P = [], [], []
correct, total = 0, 0
loss_accum = 0.0

with torch.no_grad():
    for inputs, _, labels, _dils in val_loader:
        x = inputs.squeeze().to(device).float()
        y = labels.squeeze().to(device).float()
        logits = model(x)
        loss = criterion(logits, y); loss_accum += loss.item()
        probs = torch.sigmoid(logits)
        pred = (probs > TAU).float()
        correct += (pred == y).sum().item(); total += y.numel()
        val_X.append(x.detach().cpu().numpy())
        val_Y.append(y.detach().cpu().numpy())
        val_P.append(probs.detach().cpu().numpy())

mrae_out_val_average = loss_accum / max(len(val_loader), 1)
val_acc_micro = correct / max(total, 1)

X_val_np = np.concatenate(val_X, axis=0)
Y_val_np = np.concatenate(val_Y, axis=0).astype(int)
P_val_np = np.concatenate(val_P, axis=0)

pred_val_bin = (P_val_np >= TAU).astype(int)
acc_val, tpr_val, fpr_val = summarize_binary(Y_val_np, pred_val_bin)
print("\n[VAL] threshold=%.2f  BCE=%.4f  micro-acc=%.4f  (acc=%.4f, TPR=%.3f, FPR=%.3f)"
      % (TAU, mrae_out_val_average, val_acc_micro, acc_val, tpr_val, fpr_val))

# ------------------ REAL evaluation (unchanged) ------------------
x_real = torch.tensor(kff_np, dtype=torch.float32, device=device)
with torch.no_grad():
    P_real_np = torch.sigmoid(model(x_real)).cpu().numpy()

Y_real_np = (true_labels > 0).astype(int)
pred_real_bin = (P_real_np >= TAU).astype(int)
acc_real, tpr_real, fpr_real = summarize_binary(Y_real_np, pred_real_bin)
print("*"*60)
print("REAL (N=%d), threshold=%.2f" % (Y_real_np.shape[0], TAU))
print("acc=%.4f  TPR=%.3f  FPR=%.3f" % (acc_real, tpr_real, fpr_real))
print("*"*60)

# ------------------ pick & plot failure/ambiguous (unchanged; S5 components) ------------------
def pick_unique_cases(X, Y, P, tau, split_tag, ppm=PPM_GRID, names=CLASS_NAMES):
    picks = [pick_worst_fn(Y, P, tau), pick_worst_fp(Y, P, tau), pick_closest_to_tau(Y, P, tau)]
    seen, unique = set(), []
    for sel in picks:
        if sel is None: continue
        i, c, tag = sel
        if i in seen: continue
        seen.add(i)
        unique.append((i, c, tag))
    for k, (i, c, tag) in enumerate(unique, 1):
        fname = f"{SAVE_DIR}/{split_tag}_{tag.replace(' ','_')}.png"
        plot_case(X[i], Y[i], P[i], tau=tau, class_names=names, ppm=PPM_GRID, fname=fname)
    return unique

# VAL/REAL examples
pick_unique_cases(X_val_np, Y_val_np, P_val_np, TAU, split_tag="VAL")
pick_unique_cases(kff_np.astype(np.float32), Y_real_np, P_real_np, TAU, split_tag="REAL",
                  ppm=PPM_GRID, names=CLASS_NAMES)
print(f"Saved failure/ambiguous figures to: {SAVE_DIR}/")


# ========================== NEW: S6 generation using the same plot_case ==========================
# Save each REAL sample panel with the same style (no per-panel title), then tile into 4x4.
import matplotlib.image as mpimg

# 1) save individual REAL panels
real_panel_paths = []
for i in range(Y_real_np.shape[0]):
    fpath = os.path.join(SAVE_DIR, f"REAL_{i+1:02d}.png")
    # empty title to avoid in-panel text (caption will describe)
    plot_case(kff_np[i], Y_real_np[i], P_real_np[i], tau=TAU,
              class_names=CLASS_NAMES, ppm=PPM_GRID, title="", fname=fpath)
    real_panel_paths.append(fpath)

# 2) tile into a 4x4 S4 grid (A–P labels)
letters = [chr(ord('A')+k) for k in range(min(16, len(real_panel_paths)))]
fig, axs = plt.subplots(4, 4, figsize=(12, 12))
for ax, f, lab in zip(axs.flat, real_panel_paths, letters):
    ax.imshow(mpimg.imread(f))
    ax.set_title(lab, loc="left", fontsize=14, fontweight="bold")
    ax.axis("off")
# hide any unused panels if <16
for j in range(len(real_panel_paths), 16):
    axs.flat[j].axis("off")

plt.tight_layout()
plt.savefig("Figure_S6_REAL_gallery_grid.pdf", bbox_inches="tight", dpi=300)
plt.savefig("Figure_S6_REAL_gallery_grid.png", bbox_inches="tight", dpi=300)
print("Saved S4 to Figure_S4_REAL_gallery_grid.[pdf|png] and individual panels to", SAVE_DIR)




# ------------------ S7 grid (unchanged) ------------------
import matplotlib.image as mpimg
files = [
    "si_figs/REAL_Worst_false_negative.png",
    "si_figs/REAL_Worst_false_positive.png",
    "si_figs/REAL_Closest-to-τ_ambiguous.png",
    "si_figs/VAL_Worst_false_negative.png",
    "si_figs/VAL_Worst_false_positive.png",
    "si_figs/VAL_Closest-to-τ_ambiguous.png",
]
fig, axs = plt.subplots(2, 3, figsize=(10, 6))
for ax, f, title in zip(axs.flat, files, list("ABCDEF")):
    ax.imshow(mpimg.imread(f))
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.axis("off")
plt.tight_layout()
plt.savefig("Figure_S7_misclassified_ambiguous_grid.pdf", bbox_inches="tight", dpi=300)
plt.savefig("Figure_S7_misclassified_ambiguous_grid.png", bbox_inches="tight", dpi=300)




############################   classical chemometric analysis
# ==========================
import numpy as np, torch, gc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
from sklearn.utils import check_random_state

# --------- Data (as before) ----------
S_sim = sim["S"]            # [N_sim, L], float32
Y_sim = sim["mask"]         # [N_sim, C], int
X_real = kff_np.astype(np.float32)
Y_real = (true_labels > 0).astype(np.int32)

train_idx = np.array(train_set.indices)
val_idx   = np.array(val_set.indices)

def to_np(x): return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
X_sim_all = to_np(S_sim).astype(np.float32)
Y_sim_all = to_np(Y_sim).astype(np.int32)

X_sim_tr, Y_sim_tr = X_sim_all[train_idx], Y_sim_all[train_idx]
X_sim_val, Y_sim_val = X_sim_all[val_idx], Y_sim_all[val_idx]

# --------- Utils (as before) ----------
def proba_ovr(estimator, X):
    P = estimator.predict_proba(X)
    if isinstance(P, (list, tuple)): P = np.column_stack(P)
    else: P = np.asarray(P)
    return P.astype(np.float32)

def ece_binary_multilabel(Y_true, Y_prob, n_bins=15):
    t, p = np.asarray(Y_true).ravel(), np.asarray(Y_prob).ravel()
    bins = np.linspace(0,1,n_bins+1); ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if m.any(): ece += m.mean() * abs(t[m].mean() - p[m].mean())
    return float(ece)

def exact_match_acc(Y, Yhat):
    Y, Yhat = np.asarray(Y), np.asarray(Yhat)
    return float(np.mean(np.all(Y == Yhat, axis=1)))

def micro_metrics(Y_true, Y_prob, tau):
    Y_true = np.asarray(Y_true); Y_prob = np.asarray(Y_prob)
    Y_hat  = (Y_prob >= tau).astype(np.int32)
    return dict(
        micro_f1   = f1_score(Y_true.ravel(), Y_hat.ravel(), average='micro'),
        precision  = precision_score(Y_true.ravel(), Y_hat.ravel(), average='micro'),
        recall     = recall_score(Y_true.ravel(), Y_hat.ravel(), average='micro'),
        subset_acc = exact_match_acc(Y_true, Y_hat),
        brier      = mean_squared_error(Y_true.ravel(), Y_prob.ravel()),
        ece        = ece_binary_multilabel(Y_true, Y_prob, n_bins=15),
    )

TAU_GRID = np.arange(0.50, 0.95, 0.05)
def pick_tau_star(Y_val, P_val, grid=TAU_GRID):
    yv, pv = np.asarray(Y_val).ravel(), np.asarray(P_val).ravel()
    best_tau, best = None, -1.0
    for tau in grid:
        s = f1_score(yv, (pv >= tau).astype(int), average='micro')
        if s > best: best, best_tau = s, tau
    return best_tau

def per_class_tau_star(Y_val, P_val, grid=TAU_GRID):
    C = Y_val.shape[1]; taus = np.zeros(C, dtype=np.float32)
    for c in range(C):
        y, p = Y_val[:, c], P_val[:, c]
        best, bt = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y, (p >= t).astype(int), average='binary', zero_division=0)
            if f1 > best: best, bt = f1, t
        taus[c] = bt
    return taus

def apply_per_class_tau(P, taus): return (P >= taus[None, :]).astype(np.int32)

# --------- Config: subset fitting & safer classifiers ----------
FIT_N = 6000     # number of SIM-train samples to fit PCA/PLS; tune 4000–8000 based on RAM
RAND  = 42
rng = check_random_state(RAND)
fit_sel = rng.choice(X_sim_tr.shape[0], size=min(FIT_N, X_sim_tr.shape[0]), replace=False)

# Logistic with lbfgs (more stable than liblinear here)
def make_logreg(C):
    return OneVsRestClassifier(LogisticRegression(
        penalty='l2', C=C, class_weight='balanced',
        solver='lbfgs', max_iter=2000
    ))

# --------- PCA → Logistic (subset-fit + randomized PCA) ----------
K_GRID = (32, 64, 128, 256, 512)
C_GRID = (0.1, 0.5, 1.0, 2.0, 10.0)

# Fit scaler on subset, then PCA(randomized) on subset
sc_pca = StandardScaler()
X_fit  = sc_pca.fit_transform(X_sim_tr[fit_sel])

best_pca = None
for k in K_GRID:
    pca = PCA(n_components=k, svd_solver='randomized', random_state=RAND)
    Z_fit = pca.fit_transform(X_fit)
    # transform full SIM-val with fitted scaler+pca
    Z_tr  = pca.transform(sc_pca.transform(X_sim_tr))
    Z_val = pca.transform(sc_pca.transform(X_sim_val))

    for C in C_GRID:
        clf = make_logreg(C)
        clf.fit(Z_tr, Y_sim_tr)
        P_val = proba_ovr(clf, Z_val)
        tau_star = pick_tau_star(Y_sim_val, P_val)
        mf1 = micro_metrics(Y_sim_val, P_val, tau_star)['micro_f1']
        if (best_pca is None) or (mf1 > best_pca['score']):
            best_pca = dict(scaler=sc_pca, pca=pca, clf=clf, k=k, C=C, tau=tau_star, score=mf1)
    # memory housekeeping
    del Z_fit, Z_tr, Z_val; gc.collect()

# Evaluate PCA baseline on REAL
Z_real = best_pca['pca'].transform(best_pca['scaler'].transform(X_real))
P_real_pca = proba_ovr(best_pca['clf'], Z_real)
pca_tau70    = micro_metrics(Y_real, P_real_pca, 0.70)
pca_taustar  = micro_metrics(Y_real, P_real_pca, best_pca['tau'])
# per-class τ_c from SIM-val
Z_val_full   = best_pca['pca'].transform(best_pca['scaler'].transform(X_sim_val))
taus_pca     = per_class_tau_star(Y_sim_val, proba_ovr(best_pca['clf'], Z_val_full))
Yhat_pca_pc  = apply_per_class_tau(P_real_pca, taus_pca)
pca_perclass = dict(
    micro_f1   = f1_score(Y_real.ravel(), Yhat_pca_pc.ravel(), average='micro'),
    precision  = precision_score(Y_real.ravel(), Yhat_pca_pc.ravel(), average='micro'),
    recall     = recall_score(Y_real.ravel(), Yhat_pca_pc.ravel(), average='micro'),
    subset_acc = exact_match_acc(Y_real, Yhat_pca_pc),
    brier      = mean_squared_error(Y_real.ravel(), P_real_pca.ravel()),
    ece        = ece_binary_multilabel(Y_real, P_real_pca, n_bins=15),
)

# --------- PLS-DA (subset-fit PLS2; optional pre-PCA if RAM tight) ----------
M_GRID = (16, 32, 64, 96, 128)


best_pls = None
for m in M_GRID:
    # Fit scaler + PLS on subset
    sc_pls = StandardScaler()
    X_fit  = sc_pls.fit_transform(X_sim_tr[fit_sel])
    pls    = PLSRegression(n_components=m, scale=False)
    pls.fit(X_fit, Y_sim_tr[fit_sel])

    # Transform full SIM train/val
    Z_tr = pls.transform(sc_pls.transform(X_sim_tr))
    Z_val = pls.transform(sc_pls.transform(X_sim_val))

    for C in C_GRID:
        clf = make_logreg(C)
        clf.fit(Z_tr, Y_sim_tr)
        P_val = proba_ovr(clf, Z_val)
        tau_star = pick_tau_star(Y_sim_val, P_val)
        mf1 = micro_metrics(Y_sim_val, P_val, tau_star)['micro_f1']
        if (best_pls is None) or (mf1 > best_pls['score']):
            best_pls = dict(scaler=sc_pls, pls=pls, clf=clf, m=m, C=C, tau=tau_star, score=mf1)
    del Z_tr, Z_val; gc.collect()

# Evaluate PLS baseline on REAL
Z_real_pls   = best_pls['pls'].transform(best_pls['scaler'].transform(X_real))
P_real_pls   = proba_ovr(best_pls['clf'], Z_real_pls)
pls_tau70    = micro_metrics(Y_real, P_real_pls, 0.70)
pls_taustar  = micro_metrics(Y_real, P_real_pls, best_pls['tau'])
# per-class τ_c from SIM-val
Z_val_pls    = best_pls['pls'].transform(best_pls['scaler'].transform(X_sim_val))
taus_pls     = per_class_tau_star(Y_sim_val, proba_ovr(best_pls['clf'], Z_val_pls))
Yhat_pls_pc  = apply_per_class_tau(P_real_pls, taus_pls)
pls_perclass = dict(
    micro_f1   = f1_score(Y_real.ravel(), Yhat_pls_pc.ravel(), average='micro'),
    precision  = precision_score(Y_real.ravel(), Yhat_pls_pc.ravel(), average='micro'),
    recall     = recall_score(Y_real.ravel(), Yhat_pls_pc.ravel(), average='micro'),
    subset_acc = exact_match_acc(Y_real, Yhat_pls_pc),
    brier      = mean_squared_error(Y_real.ravel(), P_real_pls.ravel()),
    ece        = ece_binary_multilabel(Y_real, P_real_pls, n_bins=15),
)

# --------- Print summary ----------
def r3(d): return {k:(round(float(v),3) if isinstance(v,(int,float,np.floating)) else v) for k,v in d.items()}
print(f"\n[PCA→LogReg] k={best_pca['k']} C={best_pca['C']} τ*={best_pca['tau']:.2f} (subset-fit N={len(fit_sel)})")
print("  τ=0.70   :", r3(pca_tau70))
print("  τ*=SIM   :", r3(pca_taustar))
print("  per-class:", r3(pca_perclass))

print(f"\n[PLS-DA→LogReg] m={best_pls['m']} C={best_pls['C']} τ*={best_pls['tau']:.2f} (subset-fit N={len(fit_sel)})")
print("  τ=0.70   :", r3(pls_tau70))
print("  τ*=SIM   :", r3(pls_taustar))
print("  per-class:", r3(pls_perclass))




# Already have P_real_pca / P_real_pls
pca_tau50   = micro_metrics(Y_real, P_real_pca, 0.50)
pls_tau50   = micro_metrics(Y_real, P_real_pls, 0.50)
# (ECE/Brier are already included by micro_metrics)
print("  τ=0.50   :", r3(pca_tau50))
print("  τ=0.50   :", r3(pls_tau50))



###                                simimarity check

X_sim = S_sim
# Y_sim = Y_sim

X_real = kff_np.astype(np.float32)
Y_real = (true_labels > 0).astype(np.int32)
# -------- Inputs you already have --------

#                                   Figure S4 — PCA overlap (fit on SIM only)
# Inputs expected:
#   X_sim  : (n_sim, L)  float32 — preprocessed same as DL inference
#   X_real : (n_real, L) float32 — preprocessed same as DL inference

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# ==== font sizes (old → new): labels 12→13, title 14→15, legend 10→11, ticks 10→11 ====
FS_LABEL  = 13
FS_TITLE  = 15
FS_LEGEND = 11
FS_TICK   = 11

# X_sim, X_real provided by you
sc = StandardScaler()
X_sim_z  = sc.fit_transform(np.asarray(X_sim,  dtype=np.float32))
X_real_z = sc.transform      (np.asarray(X_real, dtype=np.float32))

pca = PCA(n_components=10, svd_solver="randomized", random_state=42)
Z_sim  = pca.fit_transform(X_sim_z)
Z_real = pca.transform(X_real_z)

rng = np.random.default_rng(42)
sel = rng.choice(Z_sim.shape[0], size=min(6000, Z_sim.shape[0]), replace=False)
Zs = Z_sim[sel]

def jitter_points(Z, sx=0.4, sy=0.1, seed=7):
    r = np.random.default_rng(seed)
    J = Z.copy()
    J[:,0] += r.normal(0, sx, size=J.shape[0])
    J[:,1] += r.normal(0, sy, size=J.shape[0])
    return J

Zr_12 = jitter_points(Z_real[:,[0,1]])
Zr_23 = jitter_points(Z_real[:,[1,2]])

ve = pca.explained_variance_ratio_
def axlabel(i): return f"PC{i+1} ({ve[i]*100:.1f}%)"

plt.figure(figsize=(10.5,4.8), dpi=150)

# PC1–PC2
ax = plt.subplot(1,2,1)
ax.scatter(Zs[:,0], Zs[:,1], s=6, alpha=0.25, label="SIM", linewidths=0)
ax.scatter(Zr_12[:,0], Zr_12[:,1], s=48, facecolors='none', edgecolors='tab:orange', linewidths=1.6, label="REAL (jittered)")
ax.set_xlabel(axlabel(0), fontsize=FS_LABEL); ax.set_ylabel(axlabel(1), fontsize=FS_LABEL)
# ax.set_title("PCA (PC1–PC2)", fontsize=FS_TITLE)
ax.tick_params(labelsize=FS_TICK)
ax.legend(frameon=False, loc="upper right", markerscale=3, prop={'size': FS_LEGEND})

# PC2–PC3
ax = plt.subplot(1,2,2)
ax.scatter(Zs[:,1], Zs[:,2], s=6, alpha=0.25, label="SIM", linewidths=0)

ax.scatter(Zr_23[:,0], Zr_23[:,1], s=48, facecolors='none', edgecolors='tab:orange', linewidths=1.6, label="REAL (jittered)")
ax.set_xlabel(axlabel(1), fontsize=FS_LABEL); ax.set_ylabel(axlabel(2), fontsize=FS_LABEL)
# ax.set_title("PCA (PC2–PC3)", fontsize=FS_TITLE)
ax.tick_params(labelsize=FS_TICK)
ax.legend(frameon=False, loc="upper right", markerscale=3, prop={'size': FS_LEGEND})

plt.tight_layout()
plt.savefig("Figure_S4_PCA_overlap.pdf", bbox_inches="tight")
plt.savefig("Figure_S4_PCA_overlap.png", bbox_inches="tight", dpi=300)
print("Saved Figure_S4_PCA_overlap.(pdf|png)")




#                                     Figure S5 — UMAP overlap (fit on SIM only)
# Figure S5. UMAP overlap between simulated (SIM) and real (REAL) spectra. 
# UMAP was fit on SIM only (no leakage), then used to transform REAL into the learned embedding. 
# REAL points embed within the SIM cloud rather than forming a separate island.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

# ==== font sizes ====
FS_LABEL  = 13
FS_TITLE  = 15
FS_LEGEND = 11
FS_TICK   = 11

# Standardize on SIM only
sc = StandardScaler(with_mean=True, with_std=True)
X_sim_z  = sc.fit_transform(np.asarray(X_sim,  dtype=np.float32))
X_real_z = sc.transform      (np.asarray(X_real, dtype=np.float32))

EMBEDDER_NAME = None
try:
    from umap import UMAP
    um = UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42, transform_seed=42)
    Z_sim_2d  = um.fit_transform(X_sim_z)
    Z_real_2d = um.transform(X_real_z)
    EMBEDDER_NAME = "UMAP"
except Exception:
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(42)
    sel = rng.choice(X_sim_z.shape[0], size=min(8000, X_sim_z.shape[0]), replace=False)
    X_sim_plot  = X_sim_z[sel]
    X_real_plot = X_real_z
    X_all = np.vstack([X_sim_plot, X_real_plot])
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42)
    Z_all = tsne.fit_transform(X_all)
    Z_sim_2d  = Z_all[:len(X_sim_plot)]
    Z_real_2d = Z_all[len(X_sim_plot):]
    EMBEDDER_NAME = "t-SNE"

plt.figure(figsize=(6.4,5.4), dpi=150)
plt.scatter(Z_sim_2d[:,0], Z_sim_2d[:,1], s=4, alpha=0.30, label="SIM", linewidths=0)

plt.scatter(Z_real_2d[:,0], Z_real_2d[:,1], s=36, marker='x', linewidths=1.2, alpha=0.95, label="REAL")

plt.xlabel(f"{EMBEDDER_NAME}-1", fontsize=FS_LABEL)
plt.ylabel(f"{EMBEDDER_NAME}-2", fontsize=FS_LABEL)
# plt.title(f"{EMBEDDER_NAME} overlap (fit on SIM{' ; transform REAL' if EMBEDDER_NAME=='UMAP' else ' + REAL for viz'})", fontsize=FS_TITLE)
plt.tick_params(labelsize=FS_TICK)
plt.legend(frameon=False, loc="best", markerscale=3, prop={'size': FS_LEGEND})
plt.tight_layout()

fname = f"Figure_S5_{EMBEDDER_NAME}_overlap"
plt.savefig(fname + ".pdf", bbox_inches="tight")
plt.savefig(fname + ".png", bbox_inches="tight", dpi=300)
print("Saved:", fname + ".pdf", "and", fname + ".png")





## check the correlations
import numpy as np

# X_sim, X_real: preprocessed like DL inference (same grid/normalization)
# For speed you may subsample SIM to, say, 10k rows.
rng = np.random.default_rng(42)
sub = rng.choice(len(X_sim), size=min(10000, len(X_sim)), replace=False)
S = X_sim[sub].astype(np.float32)
R = X_real.astype(np.float32)

# center and L2-normalize rows -> cosine ≈ Pearson r on centered data
S0 = S - S.mean(axis=1, keepdims=True)
S0 /= np.linalg.norm(S0, axis=1, keepdims=True) + 1e-12
corr_max = []
for r in R:
    r0 = r - r.mean()
    r0 /= np.linalg.norm(r0) + 1e-12
    corr_max.append((S0 @ r0).max())  # max Pearson-like corr vs SIM pool
corr_max = np.array(corr_max)
print("median:", float(np.median(corr_max)))
print("IQR:", (float(np.percentile(corr_max,25)), float(np.percentile(corr_max,75))))






