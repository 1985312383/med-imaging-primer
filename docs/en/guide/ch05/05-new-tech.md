---
title: "5.5 New Paradigms: SAM & Generative AI"
description: Foundation models and generative priors in medical imaging workflows
---

# 5.5 New Paradigms: SAM & Generative AI

In recent years, medical imaging AI has been shifting from “train one model per task” toward **foundation models** and **generative priors**.

---

## SAM in medical imaging

Common uses:

- interactive annotation acceleration (point/box prompts)
- semi-automatic labeling pipelines (model suggestion + human correction)
- bootstrapping downstream specialist models (e.g., nnU-Net) with cheaper labels

:::: warning ⚠️ Domain gap matters
SAM is largely trained on natural images. For medical images, low-contrast boundaries, tiny lesions, and 3D volumes often require adaptation and careful validation.
::::

---

## Generative AI: denoising, reconstruction, and synthesis

Typical applications:

- low-dose CT denoising / artifact reduction
- accelerated MRI reconstruction
- data augmentation / long-tail synthesis

:::: warning ⚠️ Hallucination risk
In medical imaging, “looks plausible” is not enough. Generative models must be evaluated for false positives/negatives and integrated with QA and uncertainty-aware checks.
::::


