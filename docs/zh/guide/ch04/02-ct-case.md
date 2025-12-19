---
title: 4.2 案例一：CT 正弦图回放与重建
description: 从 Phantom 到 sinogram，再到 FBP 重建与可视化
---

# 4.2 案例一：CT 正弦图回放与重建

本节做一个“最小闭环”的 CT 实验：**生成 Phantom → Radon 得到正弦图（sinogram）→ FBP 重建 → 可视化对比**。目标不是追求极致指标，而是把数据流跑通。

---

## 1. 生成 Phantom

这里用 `skimage` 自带的 Shepp-Logan Phantom（经典 CT 测试图）。

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom

img = shepp_logan_phantom()
plt.imshow(img, cmap="gray")
plt.title("Phantom")
plt.axis("off")
plt.show()
```

---

## 2. 正弦图（sinogram）回放

```python
from skimage.transform import radon

angles = np.linspace(0., 180., max(img.shape), endpoint=False)
sino = radon(img, theta=angles, circle=True)

plt.imshow(sino, cmap="gray", aspect="auto")
plt.title("Sinogram")
plt.xlabel("Angle index")
plt.ylabel("Detector index")
plt.show()
```

---

## 3. FBP 重建

```python
from skimage.transform import iradon

recon = iradon(sino, theta=angles, filter_name="ramp", circle=True)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
ax[1].imshow(recon, cmap="gray"); ax[1].set_title("FBP (ramp)"); ax[1].axis("off")
plt.tight_layout()
plt.show()
```

---

## 4. 小结

你已经跑通了 CT 的最简实验链路：图像域 ↔ 投影域 ↔ 重建。下一步可以把角度数减少（稀疏视角）、加入噪声、换不同滤波器，然后在 `4.4` 用 PSNR/SSIM 做定量对比。


