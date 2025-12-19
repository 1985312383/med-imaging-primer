---
title: 3.1 Analytic Reconstruction (FBP/FFT)
description: From projection geometry to FBP/FDK (and FFT as an analytic inverse)
---

# 3.1 Analytic Reconstruction (FBP/FFT)

Analytic reconstruction provides fast, formula-based (or transform-based) solutions under assumptions. Typical examples:

- **CT**: FBP / FDK
- **MRI**: FFT-based reconstruction (Cartesian sampling)

This page focuses on CT: geometry → Radon transform → FBP → FDK.

---

## Core idea (CT)

CT measures line integrals of attenuation. Under ideal sampling, we can recover the image via the inverse Radon transform, implemented efficiently with filtering + backprojection.

---

## FBP (Filtered Backprojection)

FBP uses:

1. **Filtering** each projection (ramp / Hann / etc.)
2. **Backprojection** over angles

Compact form:

$$
f(x, y) = \int_0^{\pi} [Rf(\theta, s) * h(s)] \bigg|_{s = x\cos\theta + y\sin\theta}\, d\theta
$$

---

## FDK (Cone-beam CT)

FDK extends FBP to cone-beam geometry using:

- geometric weighting
- 1D filtering (often FFT-based)
- 3D backprojection

---

## Next

- Iterative reconstruction: `/en/guide/ch03/02-iterative-recon`
- Deep learning reconstruction: `/en/guide/ch03/03-dl-recon`


