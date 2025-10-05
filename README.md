# Single-Pass CNN–Transformer for Multi-Label ¹H NMR Flavor Mixture Identification

This repository trains a multi-label classifier to decompose formulated 1H NMR spectra into 13 reference components.
Training data are simulated from pure references with realistic artifacts (axis warp/stretch, baseline/ripple, noise,
dilution, polarity flips). 
We provide three model variants for ablation:
- hybrid (default): CNN downsampling → Transformer → classifier
- cnn: CNN-only
- transformer: Transformer-only (patch embedding to 64 tokens)
 
The file with a small label correction for two reported test mixtures are saved in 'Formulated_Flavor_Ratios_new.csv'.
