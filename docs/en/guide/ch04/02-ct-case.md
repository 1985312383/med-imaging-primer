---
title: "4.2 Case 1: CT Sinogram Playback & Reconstruction"
description: Phantom → sinogram → FBP reconstruction and visualization
---

# 4.2 Case 1: CT Sinogram Playback & Reconstruction

This is a minimal end-to-end CT demo: **phantom → sinogram (Radon) → FBP reconstruction**.

---

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

img = shepp_logan_phantom()
angles = np.linspace(0., 180., max(img.shape), endpoint=False)
sino = radon(img, theta=angles, circle=True)
recon = iradon(sino, theta=angles, filter_name="ramp", circle=True)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("Phantom"); ax[0].axis("off")
ax[1].imshow(sino, cmap="gray", aspect="auto"); ax[1].set_title("Sinogram"); ax[1].axis("off")
ax[2].imshow(recon, cmap="gray"); ax[2].set_title("FBP"); ax[2].axis("off")
plt.tight_layout()
plt.show()
```


