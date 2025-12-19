---
title: "4.3 Case 2: MRI k-Space Imaging Lab"
description: Understand the FFT relationship between image space and k-space with a toy experiment
---

# 4.3 Case 2: MRI k-Space Imaging Lab

MRI raw data lives in **k-space**. This toy example demonstrates the 2D FFT/IFFT relationship and the effect of crude undersampling.

---

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

img = resize(shepp_logan_phantom(), (256, 256), anti_aliasing=True)
k = np.fft.fftshift(np.fft.fft2(img))

mask = np.zeros_like(img, dtype=bool)
mask[::4, :] = True
k_us = k * mask

recon = np.real(np.fft.ifft2(np.fft.ifftshift(k)))
recon_us = np.real(np.fft.ifft2(np.fft.ifftshift(k_us)))

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("Image"); ax[0].axis("off")
ax[1].imshow(np.log1p(np.abs(k_us)), cmap="gray"); ax[1].set_title("Undersampled k"); ax[1].axis("off")
ax[2].imshow(recon_us, cmap="gray"); ax[2].set_title("Undersampled recon"); ax[2].axis("off")
plt.tight_layout()
plt.show()
```


