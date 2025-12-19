---
title: "4.4 Metrics: PSNR & SSIM"
description: Quantitatively evaluate reconstruction results and understand metric limitations
---

# 4.4 Metrics: PSNR & SSIM

PSNR and SSIM are common baselines for measuring similarity between a reconstruction and a reference.

---

```python
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

img = shepp_logan_phantom()
angles = np.linspace(0., 180., max(img.shape), endpoint=False)
sino = radon(img, theta=angles, circle=True)
recon = iradon(sino, theta=angles, filter_name="ramp", circle=True)

psnr = peak_signal_noise_ratio(img, recon, data_range=img.max() - img.min())
ssim = structural_similarity(img, recon, data_range=img.max() - img.min())

print("PSNR:", psnr)
print("SSIM:", ssim)
```

:::: warning ⚠️ Don’t optimize metrics blindly
In medical imaging, better PSNR/SSIM does not always mean better clinical utility. Combine metrics with visual checks and downstream task performance where possible.
::::


