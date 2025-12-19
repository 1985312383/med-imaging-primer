---
title: 4.4 质量评估：PSNR与SSIM实测
description: 用 PSNR/SSIM 对重建结果进行定量评估，并理解指标的局限性
---

# 4.4 质量评估：PSNR与SSIM实测

在重建实验里，“看起来更清晰”并不等于“更接近真值”。本节用两个常用指标：

- **PSNR**：基于 MSE 的信噪比指标，偏向像素误差
- **SSIM**：结构相似性指标，更贴近人眼对结构的敏感度

来对结果做定量比较。

---

## 1. PSNR

给定参考图像 \(I\) 与重建图像 \(\hat I\)，均方误差：

$$
\mathrm{MSE} = \frac{1}{N}\sum (I - \hat I)^2
$$

PSNR：

$$
\mathrm{PSNR} = 10\log_{10}\left(\frac{MAX^2}{\mathrm{MSE}}\right)
$$

其中 \(MAX\) 是像素最大值（常见为 1 或 255）。

---

## 2. SSIM

SSIM 更关注局部亮度、对比度与结构的相似性。实践中直接使用库函数即可。

---

## 3. 实测代码（以 CT FBP 为例）

```python
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

img = shepp_logan_phantom()
angles = np.linspace(0., 180., max(img.shape), endpoint=False)
sino = radon(img, theta=angles, circle=True)

# baseline
recon = iradon(sino, theta=angles, filter_name="ramp", circle=True)

psnr = peak_signal_noise_ratio(img, recon, data_range=img.max() - img.min())
ssim = structural_similarity(img, recon, data_range=img.max() - img.min())

print("PSNR:", psnr)
print("SSIM:", ssim)
```

---

## 4. 指标的局限性

:::: warning ⚠️ 不要“唯指标论”
PSNR/SSIM 是非常有用的 baseline，但在医学影像里仍可能出现：
- 指标更好，但病灶边界被抹平
- 指标相近，但下游分割/检测性能差异很大

因此更稳妥的做法是：**指标 + 视觉检查 + 下游任务指标** 结合评估。
::::


