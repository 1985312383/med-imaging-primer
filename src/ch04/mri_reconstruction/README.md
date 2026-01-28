# MRI重建流程示例

本目录包含完整的MRI图像重建流程实现，涵盖K空间数据处理、欠采样策略、重建算法和质量评估。

## 文件说明

- `mri_reconstruction_pipeline.ipynb` - 完整的MRI重建Jupyter Notebook，包含所有代码和可视化
- `requirements.txt` - Python依赖包列表
- `output/` - 输出图像目录

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行Notebook

```bash
jupyter notebook mri_reconstruction_pipeline.ipynb
```

或在VS Code中直接打开 `.ipynb` 文件运行。

## 重建流程概述

本示例实现了以下MRI重建流程：

### 1. K空间基础
- K空间数据生成与可视化
- 频率域与图像域的关系
- K空间滤波技术

### 2. 欠采样策略
- 线采样（Line Undersampling）
- 中心优先采样（Center-Priority Sampling）
- 径向采样（Radial Sampling）

### 3. 重建算法
- 直接IFFT重建
- 零填充重建
- POCS迭代重建（压缩感知）

### 4. 后处理
- 高斯滤波降噪
- 非锐化掩模边缘增强

### 5. 质量评估
- PSNR（峰值信噪比）
- SSIM（结构相似性指数）
- NMSE（归一化均方误差）

## 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image_size` | 图像尺寸 | 256×256 |
| `undersampling_factor` | 欠采样因子 | 4 |
| `n_iterations` | POCS迭代次数 | 50 |
| `lambda_reg` | 正则化参数 | 0.01 |

## 输出结果

运行完成后将在 `output/` 目录生成以下可视化图像：

1. `01_kspace_basics.png` - K空间基础可视化
2. `02_sampling_masks.png` - 采样掩模对比
3. `03_direct_ifft.png` - 直接IFFT重建结果
4. `04_zero_filled.png` - 零填充重建结果
5. `05_pocs_iterations.png` - POCS迭代收敛过程
6. `06_reconstruction_comparison.png` - 重建方法对比
7. `07_postprocessing.png` - 后处理效果
8. `08_quality_metrics.png` - 质量指标对比
9. `09_kspace_filtering.png` - K空间滤波效果

## 参考

- 预处理方法：[ch02/02-mri-preprocessing.md](../../docs/zh/guide/ch02/02-mri-preprocessing.md)
- 重建技术：[ch03/02-iterative-recon.md](../../docs/zh/guide/ch03/02-iterative-recon.md)
- 完整案例：[ch04/03-mri-case.md](../../docs/zh/guide/ch04/03-mri-case.md)

## 许可证

MIT License
