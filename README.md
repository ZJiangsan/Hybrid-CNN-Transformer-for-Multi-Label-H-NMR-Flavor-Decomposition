# Hybrid-CNN-Transformer-for-Multi-Label-H-NMR-Flavor-Decomposition

This repository trains a multi-label classifier to decompose formulated 1H NMR spectra into 13 reference components.
Training data are simulated from pure references with realistic artifacts (axis warp/stretch, baseline/ripple, noise,
dilution, polarity flips). We provide three model variants for ablation:
- hybrid (default): CNN downsampling → Transformer → classifier
- cnn: CNN-only
- transformer: Transformer-only (patch embedding to 64 tokens)
 
The script can also apply a small label correction for two reported test mixtures (see "Label correction" below).
 
---
Data layout
The repository expects the following structure:
 
.
├── train_nmr_decomp.py
├── README.md
├── Formulated_Flavor_Ratios_new.csv        { 16×13 (original labels, index_col=0) }
└── DeepMID/ (external)                      { referenced, not redistributed }
    └── data/
        ├── plant_flavors/                   { 13 reference spectra (Bruker) }
        └── known_formulated_flavors/        { 16 real formulations (Bruker) }
 
DeepMID dependency (no license → link only)
We do not redistribute code/data from DeepMID (https://github.com/yfWang01/DeepMID).
The script will auto-clone DeepMID into ./vendor/DeepMID on first run (or you can provide your own checkout).
You can disable auto-clone with --no-clone and follow the on-screen instructions.
 
---
Installation
 
# create env (recommended)
conda create -n nmrdecomp python=3.10 -y
conda activate nmrdecomp
 
# install deps
pip install numpy pandas scipy scikit-learn torch tqdm
 
---
Quick start
 
# Default (hybrid) — will auto-clone DeepMID if missing
python train_nmr_decomp.py   --data-root DeepMID/data   --labels-csv Formulated_Flavor_Ratios_new.csv   --out-dir outputs_hybrid
 
Ablations:
 
# CNN-only (expects T==64 after the CNN stack)
python train_nmr_decomp.py --arch cnn --out-dir outputs_cnn
 
# Transformer-only (patch embedding creates exactly 64 tokens)
python train_nmr_decomp.py --arch transformer --out-dir outputs_transformer
 
Key flags:
--arch {hybrid,cnn,transformer}   Model variant (default: hybrid)
--batch-size INT                  Default: 100
--lr FLOAT                        Default: 1e-3
--epochs INT                      Default: 200 (early stopping enabled)
--patience INT                    Default: 40
--eval-threshold FLOAT            Default: 0.70 (for diagnostic binarization)
--no-eval-real                    Disable per-epoch evaluation on the 16 real spectra
--deepmid-url URL                 Upstream URL (default: yfWang01/DeepMID)
--deepmid-dest PATH               Local clone path (default: vendor/DeepMID)
--no-clone                        Do not auto-clone DeepMID (error if not found)
--label-overrides PATH.json       Optional per-mixture label corrections (see below)
 
---
Label correction (optional but documented)
 
Context.
During cross-referencing with the DeepMID study, we identified two inconsistencies in the reported labels of test mixtures.
Based on spectral overlays and mixture reconstruction, mixture 1 is most consistent with components {3,8}, and mixture 2 with components {3,5}.
We evaluate with these corrections for accurate ground-truth comparison, while retaining the originals for transparency.
 
How to reproduce.
Create a tiny JSON file (component indices are 1-based):
{
  "mixture_1": [3, 8],
  "mixture_2": [3, 5]
}
 
Save as label_overrides.json, then run:
python train_nmr_decomp.py   --labels-csv Formulated_Flavor_Ratios_new.csv   --label-overrides label_overrides.json   --out-dir outputs_corrected
 
Notes:
- Keys can be either the CSV index labels (e.g., "mixture_1"), or 0-based integer row indices as strings/numbers (e.g., "0" or 0).
- Values are arrays of 1-based component IDs (1..13). The script converts to a 13-length binary vector.
 
We recommend reporting both original and corrected results.
 
---
Outputs
- outputs_*/cnnTrans_NMR_best_<arch>.pth — best checkpoint by validation loss
- outputs_*/train_losses_<arch>.npy, val_losses_<arch>.npy — loss curves
- outputs_*/early_stop_summary_<arch>.json — summary (acc/loss/seed/epochs)
 
---
Reproducibility
- Deterministic seeding for Python, NumPy, and PyTorch (where supported).
- Synthetic dataset fixed by --seed (change to regenerate).
- Same training loop and hyperparameters across ablations.
 
---
License
This repository links to DeepMID but does not redistribute their code/data.
Please respect upstream licensing and any usage terms.
