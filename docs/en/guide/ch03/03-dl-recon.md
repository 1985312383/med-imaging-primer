---
title: 3.3 Deep Learning Reconstruction (AI4Recon)
description: Post-processing, unrolled optimization, and end-to-end reconstruction with neural networks
---

# 3.3 Deep Learning Reconstruction (AI4Recon)

Deep learning reconstruction can be viewed as adding learnable components to analytic/iterative pipelines to improve quality and speed under challenging conditions (noise, sparse sampling, low dose).

---

## Three common paradigms

1. **Post-processing**: analytic reconstruction → neural denoiser/artifact remover
2. **Unrolled/learned iterative**: unfold optimization (ADMM, primal-dual) into a network
3. **End-to-end**: learn mapping from measurement domain to image domain

---

## Practical caveats

- generalization across scanners/sites
- interpretability and safety (avoid “hallucinated” anatomy)
- clinical validation and standardization

---

## Next

Try an end-to-end mini pipeline in Chapter 4 (case studies).


