#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 12:36:02 2025

@author: nibio
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn

from sklearn.metrics import confusion_matrix

# -----------------------------------------------------------------------------
# Project-specific imports
from DeepMID.readBruker import read_bruker_hs_base

# -----------------------------------------------------------------------------
# 
print("Reading Bruker data ...")
plant_flavors = read_bruker_hs_base('DeepMID/data/plant_flavors', False, True, False)
known_formulated_flavors = read_bruker_hs_base('DeepMID/data/known_formulated_flavors', False, True, False)

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
##
# Load labels for real 16 (13 classes)
true_labels = pd.read_csv('Formulated_Flavor_Ratios_new.csv', index_col=0).values


# Model
class NMR_decomp(nn.Module):
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

## inference

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NMR_decomp().to(device).float()
best_model = "unsupervised_Decomp_learning_cnnTrans_NMR_val_EStop.pth"
if os.path.isfile(best_model):
    ckpt_model = torch.load(best_model, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_model['state_dict'])
    model.eval()
    x_real = torch.tensor(kff_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x_real)).detach().cpu().numpy()
    pred_binary = (probs >= float(0.70)).astype(int)
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
    print(f"current Accuracy:  {accuracy:.3f}")
    print(f"current Recall:    {tpr:.3f}")
    print(f"current FPR:       {fpr:.3f}")
    print("*"*60)




