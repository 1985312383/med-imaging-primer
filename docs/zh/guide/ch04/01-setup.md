---
title: 4.1 实验环境搭建（ODL/TIGRE）
description: 从零搭建可复现的重建实验环境，并验证关键依赖是否可用
---

# 4.1 实验环境搭建（ODL/TIGRE）

第4章我们把“重建”从公式带到可复现的实验。本节目标是：用尽量少的环境摩擦，把 **ODL / TIGRE（可选）/ 常用 Python 科学计算栈** 跑起来，并能在本地成功运行一个最小 demo。

> 说明：本教程兼容 Windows / Linux。Windows 上 GPU 相关依赖可能受限，建议优先跑通 CPU 版本。

---

## ✅ 1. Python 环境建议

推荐使用 Conda 创建干净环境（避免包冲突）：

```bash
conda create -n medimg python=3.10
conda activate medimg
```

安装基础依赖：

```bash
pip install numpy scipy matplotlib scikit-image
```

---

## 🧰 2. 安装 ODL（推荐）

ODL（Operator Discretization Library）非常适合教学与快速原型：

```bash
pip install odl
```

简单验证（能 import + 跑一个小算子）：

```python
import odl
space = odl.uniform_discr([-1, -1], [1, 1], (64, 64))
op = odl.IdentityOperator(space)
print(op(space.one()).shape)
```

---

## 🛰️ 3. 安装 TIGRE（可选）

TIGRE 更偏“工程化的 CBCT/CT 重建实验”。安装方式会因平台与 CUDA 环境不同而不同。

如果你暂时不想折腾 CUDA，建议先跳过 TIGRE，只用 ODL/ASTRA 跑通流程，再回头补齐。

---

## 🧪 4. 最小自检：能画出一张图

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 400)
plt.plot(x, np.sin(x))
plt.title("Environment OK")
plt.show()
```

能正常显示图形，说明 Python + 绘图库配置完成。


