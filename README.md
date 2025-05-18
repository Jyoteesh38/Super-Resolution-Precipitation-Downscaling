# Super-Resolution-Precipitation-Downscaling
This project applies Super-Resolution Deconvolutional Neural Networks with step orography (SRDN-SO) to downscale hourly precipitation from 100 km to 12.5 km over Australia.

## Key Features
- 8× spatial resolution enhancement
- Orography input at various model stages
- Distributed training with Horovod
- Evaluation with PSNR, SSIM, and spectral FFT
- **Trained on a massive dataset:**
  - Total images: 359,424
  - Training set: 286,720 samples
  - Test set: 72,704 samples
- **Scalable distributed training across 80 GPUs** on the Gadi supercomputer (NCI Australia)

This setup enables rapid and efficient training of deep learning models on large-scale high-resolution climate data.
"""
