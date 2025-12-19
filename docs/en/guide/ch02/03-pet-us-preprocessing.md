---
title: "2.3 PET/US: Attenuation Correction & Denoising"
description: Why attenuation correction matters in PET quantification and what “denoising” means in ultrasound
---

# 2.3 PET/US: Attenuation Correction & Denoising

This chapter connects two practical preprocessing topics:

- **PET attenuation correction (AC)** for quantitative imaging
- **Ultrasound noise/artifacts** and basic denoising strategies

---

## PET: why attenuation correction is essential

Gamma photons are attenuated (absorbed/scattered) inside the body, causing systematic underestimation for deeper structures.

Common approaches:

- **PET-CT**: build a μ-map from CT (and map energies to 511 keV)
- **PET-MRI**: estimate μ-map via segmentation/atlas/UTE (harder)

AC is one part of a full quantitative chain (scatter/randoms/dead-time/normalization are also common).

---

## Ultrasound: noise and artifacts

Typical issues:

- **Speckle** (coherent imaging phenomenon)
- **Shadowing / enhancement**
- **Reverberation**
- **Refraction / side lobes**

Denoising strategies:

- classical edge-preserving filters (median, bilateral, diffusion)
- learning-based/self-supervised denoisers (task-aware)


