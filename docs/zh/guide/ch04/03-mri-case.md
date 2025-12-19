---
title: 4.3 案例二：MRI K空间成像实验
description: 用 FFT/IFFT 理解 k-space 与图像域的对应关系，并做简单采样实验
---

# 4.3 案例二：MRI K空间成像实验

MRI 的“原始数据”在 **k 空间**。本节用一个最小实验理解：**图像 ↔ k-space 的二维 FFT 关系**，以及欠采样会带来什么现象。

---

## 1. 从一张图像构造 k-space

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

img = resize(shepp_logan_phantom(), (256, 256), anti_aliasing=True)

k = np.fft.fftshift(np.fft.fft2(img))
mag = np.log1p(np.abs(k))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("Image"); ax[0].axis("off")
ax[1].imshow(mag, cmap="gray"); ax[1].set_title("log|k-space|"); ax[1].axis("off")
plt.tight_layout()
plt.show()
```

---

## 2. 从 k-space 重建回图像域

```python
recon = np.fft.ifft2(np.fft.ifftshift(k))
recon = np.real(recon)

plt.imshow(recon, cmap="gray")
plt.title("IFFT reconstruction")
plt.axis("off")
plt.show()
```

---

## 3. 简单欠采样实验：行采样

下面做一个非常“粗暴”的欠采样：只保留部分 k-space 行（示意用，不代表真实采样策略）。

```python
mask = np.zeros_like(img, dtype=bool)
mask[::4, :] = True  # 每4行保留1行

k_us = k * mask
recon_us = np.real(np.fft.ifft2(np.fft.ifftshift(k_us)))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original"); ax[0].axis("off")
ax[1].imshow(np.log1p(np.abs(k_us)), cmap="gray"); ax[1].set_title("Undersampled k"); ax[1].axis("off")
ax[2].imshow(recon_us, cmap="gray"); ax[2].set_title("Undersampled recon"); ax[2].axis("off")
plt.tight_layout()
plt.show()
```

---

## 4. 小结

这节的核心是把“k-space 是什么”从概念落到可视化：欠采样会带来混叠/伪影。更真实的 MRI 欠采样策略（如中心密集、径向/螺旋）与去混叠方法，会在第3章/第5章逐步展开。


