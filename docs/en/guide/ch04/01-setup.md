---
title: 4.1 Environment Setup (ODL/TIGRE)
description: Set up a reproducible reconstruction environment and verify core dependencies
---

# 4.1 Environment Setup (ODL/TIGRE)

Goal: get a minimal, reproducible environment to run the Chapter 4 case studies.

---

## 1) Create a clean Python environment

```bash
conda create -n medimg python=3.10
conda activate medimg
pip install numpy scipy matplotlib scikit-image
```

---

## 2) Install ODL (recommended)

```bash
pip install odl
```

Quick check:

```python
import odl
space = odl.uniform_discr([-1, -1], [1, 1], (64, 64))
print(space.one().shape)
```

---

## 3) TIGRE (optional)

TIGRE installation varies across platforms/CUDA versions. If you don’t want to deal with CUDA yet, you can skip TIGRE and still complete the case studies with CPU-friendly tooling.


