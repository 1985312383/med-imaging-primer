---
title: 4.3 案例二：MRI K空间成像与重建实验
description: 完整的MRI K空间成像流程，包括欠采样策略、重建算法与质量评估
---

# 4.3 案例二：MRI K空间成像与重建实验

MRI 的原始数据在 **K空间**（频率域）。本节通过完整的实验流程，理解图像与K空间的二维FFT关系，探索不同欠采样策略对重建质量的影响，并实现多种重建算法进行对比分析。

---

## 1. K空间基础

### 1.1 图像到K空间的变换

MRI信号在K空间采集，通过二维傅里叶变换（FFT）转换到图像域：

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

# 生成测试图像
img = resize(shepp_logan_phantom(), (256, 256), anti_aliasing=True)

# 图像 → K空间
k_space = np.fft.fftshift(np.fft.fft2(img))
magnitude = np.log1p(np.abs(k_space))
phase = np.angle(k_space)
```

![K空间基础](/images/ch04/01_kspace_basics.png)

**关键观察**：
- K空间中心对应图像的低频信息（对比度）
- K空间边缘对应图像的高频信息（细节/边缘）
- 相位信息对图像重建至关重要

### 1.2 全采样重建

从完整K空间数据通过逆傅里叶变换（IFFT）重建图像：

```python
# K空间 → 图像
recon_full = np.real(np.fft.ifft2(np.fft.ifftshift(k_space)))
```

![全采样重建](/images/ch04/02_full_sampling_recon.png)

全采样重建能够完美恢复原始图像（数值误差 < 1e-10）。

---

## 2. 欠采样策略

为加速MRI扫描，常采用欠采样策略。以下是三种典型方法：

### 2.1 线采样（Line Undersampling）

每隔N行保留一行，实现简单但会引入混叠伪影：

```python
def create_line_undersampling_mask(size, factor=4):
    """线采样掩模"""
    mask = np.zeros((size, size), dtype=bool)
    mask[::factor, :] = True
    return mask
```

### 2.2 中心优先采样（Center-Priority）

保留K空间中心区域（低频信息），对边缘进行欠采样：

```python
def create_center_priority_mask(size, center_lines=32, outer_factor=4):
    """中心优先采样掩模"""
    mask = np.zeros((size, size), dtype=bool)
    center_start = (size - center_lines) // 2
    mask[center_start:center_start+center_lines, :] = True
    # 边缘欠采样
    outer_indices = np.arange(0, center_start, outer_factor)
    mask[outer_indices, :] = True
    mask[size-1:center_start+center_lines:-outer_factor, :] = True
    return mask
```

### 2.3 径向采样（Radial Sampling）

沿径向方向采样，对运动不敏感：

```python
def create_radial_mask(size, n_spokes=16):
    """径向采样掩模"""
    mask = np.zeros((size, size), dtype=bool)
    center = size // 2
    angles = np.linspace(0, np.pi, n_spokes, endpoint=False)
    for angle in angles:
        for r in range(size//2):
            x = int(center + r * np.cos(angle))
            y = int(center + r * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                mask[y, x] = True
    return mask
```

![采样掩模对比](/images/ch04/03_sampling_masks.png)

---

## 3. 欠采样伪影分析

不同欠采样策略会在重建图像中引入不同类型的伪影：

![欠采样伪影](/images/ch04/04_undersampling_artifacts.png)

**伪影特征**：
- **线采样**：沿相位编码方向的混叠（Aliasing）
- **中心优先**：相对较少的伪影，但高频信息丢失
- **径向采样**：星状伪影（Streaking Artifacts）

---

## 4. 重建算法

### 4.1 直接IFFT重建

最简单的重建方法，但欠采样时质量较差：

```python
# 应用采样掩模
k_undersampled = k_space * mask

# 直接IFFT重建
recon_direct = np.real(np.fft.ifft2(np.fft.ifftshift(k_undersampled)))
```

### 4.2 零填充重建

对未采样点进行零填充，减少混叠但会模糊：

```python
# 计算采样密度补偿
sampling_density = np.sum(mask) / mask.size

# 零填充重建
k_zero_filled = k_undersampled.copy()
k_zero_filled[~mask] = 0
recon_zero = np.real(np.fft.ifft2(np.fft.ifftshift(k_zero_filled))) / sampling_density
```

### 4.3 POCS迭代重建（压缩感知）

利用图像稀疏性进行迭代重建，显著减少伪影：

```python
def pocs_reconstruction(k_undersampled, mask, n_iterations=50, lambda_reg=0.01):
    """
    POCS (Projection Onto Convex Sets) 迭代重建
    
    参数:
        k_undersampled: 欠采样K空间数据
        mask: 采样掩模
        n_iterations: 迭代次数
        lambda_reg: 正则化参数（软阈值）
    """
    # 初始估计：零填充重建
    x = np.real(np.fft.ifft2(np.fft.ifftshift(k_undersampled)))
    
    for i in range(n_iterations):
        # 步骤1: 变换到K空间
        k = np.fft.fftshift(np.fft.fft2(x))
        
        # 步骤2: 数据一致性投影（保持采样点不变）
        k = k * ~mask + k_undersampled
        
        # 步骤3: 变换回图像域
        x = np.real(np.fft.ifft2(np.fft.ifftshift(k)))
        
        # 步骤4: 稀疏性约束（软阈值）
        x = np.sign(x) * np.maximum(np.abs(x) - lambda_reg, 0)
    
    return x
```

![POCS重建](/images/ch04/05_pocs_reconstruction.png)

**POCS算法特点**：
- 迭代过程中逐步减少伪影
- 数据一致性约束确保采样点准确
- 稀疏性约束利用图像先验信息

---

## 5. K空间滤波

在K空间应用滤波器可以控制图像特性：

```python
# 低通滤波（平滑）
def create_lowpass_filter(size, cutoff_ratio=0.3):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    return R <= cutoff_ratio

# 应用滤波
lowpass_filter = create_lowpass_filter(256, 0.3)
k_filtered = k_space * lowpass_filter
img_lowpass = np.real(np.fft.ifft2(np.fft.ifftshift(k_filtered)))
```

![K空间滤波](/images/ch04/06_kspace_filtering.png)

---

## 6. 后处理

### 6.1 高斯滤波降噪

```python
from scipy.ndimage import gaussian_filter

recon_denoised = gaussian_filter(recon_pocs, sigma=0.8)
```

### 6.2 非锐化掩模边缘增强

```python
# 高斯模糊
blurred = gaussian_filter(recon_denoised, sigma=2.0)

# 非锐化掩模
unsharp_mask = recon_denoised - blurred
recon_enhanced = recon_denoised + 0.5 * unsharp_mask
```

![后处理效果](/images/ch04/07_postprocessing.png)

---

## 7. 采样率分析

不同欠采样率对重建质量的影响：

![采样率分析](/images/ch04/08_sampling_rate_analysis.png)

**关键发现**：
- 采样率 > 25%：直接IFFT可获得可接受质量
- 采样率 10-25%：需要POCS等迭代算法
- 采样率 < 10%：重建质量显著下降

---

## 8. 方法对比与质量评估

### 8.1 重建方法对比

![方法对比](/images/ch04/09_method_comparison.png)

### 8.2 定量质量指标

| 方法 | PSNR (dB) | SSIM | NMSE (%) |
|------|-----------|------|----------|
| 直接IFFT | 16.8 | 0.42 | 45.2 |
| 零填充 | 18.5 | 0.51 | 38.7 |
| POCS (50次) | 28.3 | 0.89 | 12.1 |
| POCS + 后处理 | 29.7 | 0.92 | 10.3 |

**指标说明**：
- **PSNR**（峰值信噪比）：越高越好，>30dB为优秀
- **SSIM**（结构相似性）：0-1范围，越接近1越好
- **NMSE**（归一化均方误差）：越低越好

---

## 9. 完整代码

完整的MRI重建流程代码已整理为Jupyter Notebook：

📓 [`mri_reconstruction_pipeline.ipynb`](https://github.com/1985312383/med-imaging-primer/tree/main/src/ch04/mri_reconstruction_pipeline.ipynb)

包含：
- 所有重建算法的完整实现
- 参数调优建议
- 可视化输出
- 质量评估工具

---

## 10. 小结

本案例展示了完整的MRI K空间成像与重建流程：

1. **K空间基础**：理解频率域与图像域的关系
2. **欠采样策略**：线采样、中心优先、径向采样
3. **重建算法**：直接IFFT、零填充、POCS迭代
4. **后处理**：降噪与边缘增强
5. **质量评估**：PSNR、SSIM、NMSE指标

**最佳实践建议**：
- 采样率 ≥ 25%：使用直接IFFT或零填充
- 采样率 10-25%：使用POCS迭代重建
- 采样率 < 10%：考虑深度学习重建方法
- 始终保留K空间中心低频信息

---

## 参考

- 预处理方法：[2.2 MRI预处理](../ch02/02-mri-preprocessing.md)
- 迭代重建技术：[3.2 迭代重建](../ch03/02-iterative-recon.md)
- CT重建案例：[4.2 CT重建案例](./02-ct-case.md)
