---
title: 3.3 深度学习重建（AI4Recon）简介
description: 从后处理到可学习迭代：深度学习在重建中的典型范式与挑战
---

# 3.3 深度学习重建（AI4Recon）简介

深度学习重建（AI for Reconstruction）可以看作：在“解析/迭代”两条主线之上，引入可学习模块以提升低剂量/稀疏采样场景的质量、速度与鲁棒性。

---

## 1. 三类典型范式

### 1.1 后处理（post-processing）

先用 FBP/FDK/FFT 得到初始图像，再用 CNN/Transformer 去噪、去伪影、做超分辨率等。

代表思路：

- FBPConvNet：FBP + CNN 后处理
- RED-CNN：低剂量 CT 去噪
- GAN/扩散：重建质量增强（需谨慎评估“幻觉”风险）

### 1.2 可学习迭代（unrolling / unfolded reconstruction）

把传统优化算法（如 ADMM、Primal-Dual）展开成网络结构，让其中的步长/正则项/近端算子变成可学习模块。

代表思路：

- Learned Primal-Dual
- ADMM-Net
- MoDL（模型驱动深度网络）

### 1.3 端到端（end-to-end）

直接从投影域/ k 空间到图像域学习映射，工程上需要更强的数据覆盖与更严格的泛化验证。

---

## 2. 优势与挑战（落地视角）

**优势**
- 低剂量/稀疏采样下显著提升视觉质量与指标
- 推理速度快于传统高质量迭代

**挑战**
- 数据分布偏移与跨中心泛化
- 可解释性与可控性（尤其是“幻觉”）
- 临床验证与标准化仍需时间

---

## 3. 推荐文献（起步）

1. Jin et al., “Deep Convolutional Neural Network for Inverse Problems in Imaging”, IEEE TIP, 2017.
2. Yang et al., “DuDoNet: Dual Domain Network for CT Metal Artifact Reduction”, CVPR, 2019.

---

## 下一步

如果你要把“重建”落到可复现的小实验：建议直接进入第4章 Case Study（从环境搭建到指标评估）。


