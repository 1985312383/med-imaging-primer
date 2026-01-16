---
title: "2.2 MRI: k-Space Preprocessing"
description: Deeply understand k-space acquisition, noise characteristics, parallel imaging and motion correction in a complete preprocessing workflow
---

# 2.2 MRI: k-Space Preprocessing

> "MRI does not acquire an image directly; it acquires the 'music' of the object in the frequency domain. Understanding k-space is learning the language of that music." — Wisdom of Medical Imaging Signal Processing

When radiologists look at a crisp MR image, few realize how complex the acquisition pipeline behind it is. From the moment the MR signal is captured by the receive coils, the data starts a journey in the frequency domain—from raw sampling in k-space, through noise estimation, preparation for parallel imaging, and motion-related corrections—before an inverse Fourier transform yields the image we see.

This chapter presents the complete MRI preprocessing workflow. Unlike CT (which focuses on projection-domain corrections), MRI preprocessing centers on the k-space domain. We explain the physical meaning of k-space, its noise statistics, the mechanics of parallel imaging, and how principled preprocessing improves image quality. These topics form the foundation of MRI reconstruction and are key to optimizing diagnostic performance.

## k-Space Fundamentals (Frequency-Domain Representation)

### MRI Signal Equation and Fourier Relationship

Under a 2D assumption with spatial coordinates \(x, y\), the received MR signal with gradient encoding can be written as the Fourier transform of the transverse magnetization \(M_0(x,y)\):

\[ s(\mathbf{k}) = \iint M_0(x,y)\, e^{-i\,2\pi\,(\mathbf{k}\cdot\mathbf{r})}\, d\mathbf{r},\quad \mathbf{r}=(x,y) \]

where \(\mathbf{k}=(k_x,k_y)\) is determined by gradients and time (frequency encoding and phase encoding). Once \(s(k_x,k_y)\) is acquired, the image \(I(x,y)\) is recovered by the inverse Fourier transform (IDFT in practice):

\[ I(x,y) = \iint s(k_x,k_y)\, e^{+i\,2\pi\,(k_x x + k_y y)}\, dk_x\, dk_y. \]

### Geometric Meaning and Sampling Trajectories

- k-space coordinates \((k_x,k_y)\) represent spatial frequency content. Low-frequency samples near the center determine overall contrast; high-frequency samples at the periphery govern edges and fine details.
- The sampling interval \(\Delta k\) is inversely related to the field of view (FOV); k-space extent determines spatial resolution.
- Common trajectories: Cartesian (row-by-row), radial, spiral, EPI (single-shot or multi-shot). Non-Cartesian sampling requires regridding or NUFFT before reconstruction.

### Typical Artifacts Related to k-Space

- Undersampling/aliasing (violating Nyquist): fold-over (wrap-around) artifacts.
- Truncation of high-frequency components: edge ringing (Gibbs ringing).
- Gradient delays/nonlinearities, B0/B1 inhomogeneity, eddy currents: k-space distortions causing ghosting and blurring.

## Why k-Space Preprocessing Matters (for AI and Clinical Imaging)

- Image quality foundations: blur, unclear edges, and artifacts (aliasing, Gibbs) often originate from k-space sampling or inadequate compensation.
- Reconstruction and resampling strategy: understanding trajectories, signal models, and reconstruction is essential when customizing pipelines beyond standard DICOM images.
- Noise and artifact sources: motion, gradient imperfections, coil sensitivity changes manifest as abnormal frequency components; correcting them in k-space improves downstream segmentation/detection.
- AI compatibility: changes in sampling strategies (acceleration, parallel imaging, partial Fourier, non-Cartesian) alter noise textures/structure—account for these differences in training.

## Noise Estimation and Denoising

### Noise Statistics in MR Images

Magnitude MR images often exhibit Rician or, for multi-coil combinations, non-central chi (nc-χ) distributed noise. Noise can be spatially non-stationary due to coil sensitivity and parallel imaging reconstruction. Preprocessing operations (filtering, zero-padding, regridding) also alter noise mean and covariance.

### Non-Local Means (NLM) Denoising

Core idea: leverage self-similarity of patches across the image, not only local neighborhoods, to preserve edges while suppressing noise. A generic form:

\[ \tilde I(p) = \frac{1}{C(p)} \sum_{q} w(p,q)\, I(q), \quad w(p,q) = \exp\!\Big( - \frac{\|I(\mathcal{N}_p) - I(\mathcal{N}_q)\|^2}{h^2} \Big). \]

NLM is widely used in MRI; parameters should account for non-Gaussian, spatially varying noise.

### PCA-Based Denoising

Treat patches or voxels as high-dimensional vectors and perform principal component analysis (PCA). Suppress components dominated by noise (low-variance) and reconstruct patches from retained components. PCA denoising is effective for multi-echo, time-series (fMRI), or multi-coil data; model design must consider non-stationary noise and complex-valued signals.

## Minimal k-Space Processing Pipeline

```mermaid
graph LR
    A[Object] -->|Excitation| B[Signal Acquisition\nFreq + Phase Encoding]
    B --> C[k-Space Matrix\n s(kx, ky)]
    C --> D[Preprocessing\n Filtering / Zero-fill / Regridding]
    D --> E[Inverse FT\n Image Space]
    E --> F[Image-Domain Processing\n Denoise / Correct / Segment]
```

Key steps:
- Acquisition: gradient encoding and collecting k-space lines per phase-encode step.
- Preprocessing: zero-filling, windowing, regridding (for non-Cartesian), correcting gradient delays/nonlinearity, coil sensitivity preparation.
- Inverse FT: FFT/NUFFT to image space, followed by coil combination.
- Image-domain operations: denoising, bias-field correction, segmentation/detection.

Next:
- Reconstruction principles: Chapter 3 (`/en/guide/ch03/01-analytic-recon`)


