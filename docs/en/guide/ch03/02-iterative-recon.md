---
title: "3.2 Iterative Reconstruction (SART/OSEM)"
description: "Optimization view of reconstruction: system matrix, statistics, and regularization"
---

# 3.2 Iterative Reconstruction (SART/OSEM)

Iterative reconstruction (IR) is often preferred when assumptions behind analytic methods break (low-dose, sparse-view, more complex physics). It typically solves an optimization/statistical estimation problem instead of applying a closed-form inverse.

---

## Linear model

After discretization:

$$
Af = p
$$

where \(A\) is the system matrix/operator, \(f\) is the image, and \(p\) are measured projections.

---

## SART (algebraic)

SART updates the image iteratively using projection residuals, often providing better robustness for sparse-view data.

---

## OSEM (statistical)

For Poisson-like counts (common in emission tomography), OSEM/EM-style updates are widely used and enforce non-negativity.

---

## Regularization (L2 / TV)

Common objective forms:

$$
\min_f \ \|Af-p\|_2^2 + \lambda R(f)
$$

where \(R(f)\) could be L2 (Tikhonov) or TV, etc.

---

## Next

Deep learning reconstruction: `/en/guide/ch03/03-dl-recon`


