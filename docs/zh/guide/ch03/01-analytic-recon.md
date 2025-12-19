---
title: 3.1 解析重建（FBP/FFT）
description: 从 Radon 变换到 FBP/FDK：解析重建的核心思想与工程要点
---

# 3.1 解析重建（FBP/FFT）

解析重建（Analytic Reconstruction）强调：在一定假设下，重建问题可以通过“显式公式/快速变换”高效求解。典型代表是 **CT 的 FBP/FDK**，以及 **MRI 的 FFT（本质上也是解析反演的一部分）**。

本节以 CT 为主线讲清核心概念：**投影几何 → Radon 变换 → 滤波反投影（FBP） → 锥束扩展（FDK）**。

---

## 1. 引言

计算机断层扫描（CT）重建是指从多个角度采集的 X 射线投影数据中恢复物体二维（2D）或三维（3D）图像的过程。它将投影数据（正弦图）转换为描述组织内部线性衰减系数的空间域图像。

CT 重建在数学上基于 **Radon 变换**，该变换模拟了衰减函数的线积分如何对应投影测量值。

---

## 2. 投影几何（Projection Geometry）

### 2.1 平行束几何

在平行束系统中，每个视角的 X 射线都是平行的，每个探测器元件测量沿其对应射线的总衰减。

数学上，投影 \(p(\theta, t)\) 可以表示为：

$$
p(\theta, t) = \int_{-\infty}^{+\infty} f(x, y)\, \delta(t - x\cos\theta - y\sin\theta)\, dx\,dy
$$

### 2.2 扇束几何

扇束与平行束投影之间的关系：

$$
p_{\text{fan}}(\beta, \gamma) = p_{\text{parallel}}(\theta = \beta + \gamma, t = R \sin \gamma)
$$

### 2.3 锥束几何

在锥束系统中，光源发射的射线在横向和纵向上都发散，形成 3D 锥形。常用近似解析算法是 **FDK**。

---

## 3. Radon 变换与滤波反投影（FBP）

### 3.1 Radon 变换

对于二维图像 \(f(x,y)\)，Radon 变换 \(Rf(\theta,s)\) 表示沿角度 \(\theta\) 与偏移 \(s\) 的线积分：

$$
Rf(\theta, s) = \int f(s\cos\theta - t\sin\theta,\ s\sin\theta + t\cos\theta)\, dt
$$

### 3.2 滤波反投影（FBP）

FBP 的两个核心步骤：

1. **滤波**：对每个投影在探测器域进行高通滤波（Ram-Lak、Shepp-Logan、Hann 等）
2. **反投影**：把滤波后的投影沿采集路径“涂回去”，并对角度积分/求和

紧凑形式：

$$
f(x, y) = \int_0^{\pi} [Rf(\theta, s) * h(s)] \bigg|_{s = x\cos\theta + y\sin\theta}\, d\theta
$$

### 3.3 实际考虑（工程视角）

- 角度欠采样 → 条纹伪影（streak）
- 滤波器选择直接影响噪声与分辨率权衡
- 低剂量/稀疏视角 → FBP 质量显著下降，常转向迭代重建（见 3.2）

---

## 4. FDK（锥束 CT 的近似解析重建）

FDK 将 2D FBP 扩展到 3D 锥束几何，常见于 CBCT（牙科/介入/工业检测等）。

典型步骤：

1. **几何加权**：补偿锥束发散
2. **滤波**：沿探测器行做 1D 斜坡滤波（可 FFT 加速）
3. **反投影**：把滤波结果回投影到 3D 体素网格

---

## 下一步

- 迭代重建（SART/OSEM/正则化）：`/guide/ch03/02-iterative-recon`
- 深度学习重建（AI4Recon）：`/guide/ch03/03-dl-recon`


