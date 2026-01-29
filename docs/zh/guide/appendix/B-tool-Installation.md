---
title: 附录B：工具安装
description: 介绍常见工具的安装方法。
---

# 附录B：工具安装

## 1.引言

医学影像的算法开发通常依赖多种工具链的协作，本章将介绍本教程所需的核心环境，包括 **Python、ASTRA、BART、MONAI、NiBabel** 等基础组件。我们将从基础 Python 环境配置开始，逐步说明如何安装 CT 重建库（ASTRA）、MRI 重建工具箱（BART）、深度学习框架（MONAI）与医学影像文件读写工具（NiBabel）。通过本节内容，读者可以快速搭建一个完整、可复现的医学影像开发与重建实验环境。

## 2.直接安装 python

### 2.1.window 系统

#### 2.1.1.下载 Python 安装程序

**推荐方式：从 Python 官网下载安装包**

1. 打开浏览器，访问：`https://www.python.org`
2. 顶部导航栏点击 **Downloads**
3. 页面会自动推荐适合 Windows 的版本，比如：**Download Python 3.xx**
4. 点击这个按钮，下载一个 `.exe`​ 安装程序（例如：`python-3.12.3-amd64.exe`）


#### 2.1.2. 运行安装程序（最关键的一步）

双击下载好的 `python-3.xx-amd64.exe`，按下面步骤设置：

1. 安装界面最下面有一行：

    - ✅ **务必勾选：**​**​`Add Python 3.xx to PATH`​**
    - 这一步决定以后能不能在命令行里直接使用 `python` 命令，非常重要。
2. 在 **Customize installation** 里：

    1. **Optional Features（可选功能）**   
        建议全部勾上：

        - ✅ `pip`（Python 包管理器，必须）
        - ✅ `IDLE`（自带小编辑器，可视化调试简单脚本）
        - ✅ `Documentation`
        - ✅ `Python test suite`
        - ✅ `py launcher`​  
          然后点击 **Next**
    2. ​**Advanced Options（高级选项）** ：  
        建议勾选：

        - ✅ `Install for all users`​（多用户电脑建议，路径会在 `C:\Program Files\Python3x`）
        - ✅ `Add Python to environment variables`（如果前面漏勾 PATH，这里一定要勾上）
        - ✅ `Precompile standard library`（加快首次运行速度）


#### 2.1.3. 验证 Python 是否安装成功

安装完成后，按下面步骤检查：

1. 按 `Win`​ 键，输入 `cmd`​，打开 **命令提示符（Command Prompt）**
2. 在黑窗口里输入：

    ```bash
    python --version
    ```

    或

    ```bash
    py --version
    ```
3. 如果看到类似：

    ```text
    Python 3.12.3
    ```

    说明安装成功，PATH 也生效了。

---

#### 2.1.4. 验证 pip 是否可用

在同一个命令行里输入：

```bash
pip --version
```

如果看到类似：

```text
pip 24.x from C:\Python\Python312\Lib\site-packages\pip (python 3.12)
```

说明 `pip`​ 已安装成功，可以用来安装后续的 `monai`​、`nibabel` 等包。

### 2.2.Linux 系统

#### 2.2.1. 检查系统是否已安装 Python

打开终端（Terminal），输入：

```bash
python3 --version
```

如果看到类似输出：

```text
Python 3.10.12
```

说明系统已经有 Python3 了，一般可以直接使用。

再检查 `pip` 是否存在：

```bash
pip3 --version
```

如果提示 “command not found”，说明还没安装 pip，后面会安装。

#### 2.2.2. 使用包管理器安装 Python（Ubuntu / Debian）

2.1 更新软件源

先更新一下软件包列表：

```bash
sudo apt update
```

2.2 安装 Python3 与常用依赖

执行：

```bash
sudo apt install -y python3 python3-pip python3-venv
```

说明：

- ​`python3`：Python 解释器
- ​`python3-pip`：Python 包管理工具 pip
- ​`python3-venv`：用于创建虚拟环境（不装 Conda 的情况下非常有用）

安装完成后再次确认版本：

```bash
python3 --version
pip3 --version
```

## 3.使用 conda 安装 python（推荐）

Conda 是数据科学与医学影像领域最常用的环境管理工具，能够为不同项目创建彼此独立的 Python 环境，避免包冲突。  
本节将介绍如何在 **Windows / Linux** 上安装 Conda（Miniconda）并创建干净可控的 Python 环境。

### 3.1. 下载并安装 Miniconda（推荐）

**为什么用 Miniconda 而不是 Anaconda？**

- Miniconda 更轻量，不会额外安装上千个不必要的包
- 更适合科研项目与医学影像工程
- 完全兼容 Anaconda 的功能

### 3.2. Windows 安装 Miniconda

#### **步骤 1：下载安装包**

访问官网：

👉 https://repo.anaconda.com/miniconda/


选择：

- **Miniconda3 Windows 64-bit Installer (.exe)**

#### **步骤 2：运行安装程序**

双击安装包：

1. 选择 **Just Me** 或 **All Users**
2. **重要：勾选 "Add Miniconda3 to my PATH environment variable"**

    - 如果没有勾选，也没关系，Conda 会自动加入 PATH 针对 CMD / PowerShell
3. 点击 Install


##### **步骤 3：验证安装**

打开 *Anaconda Prompt* 或 CMD：

```bash
conda --version
```

若显示：

```
conda 24.x.x
```

则安装成功。

### 3.3 Linux 安装 Miniconda

### **步骤 1：下载安装脚本**

访问官网：

👉 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

复制 Linux 安装脚本链接，例如：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### **步骤 2：运行安装脚本**

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

按提示操作：

- 阅读并同意协议
- 选择安装路径（默认即可：`~/miniconda3`）
- 选择是否初始化 conda（推荐 YES）

### **步骤 3：使 conda 生效**

如果你刚才选择了 YES，只需重新打开终端即可。  
如果选择了 NO，需要手动运行：

```bash
source ~/miniconda3/bin/activate
```

验证：

```bash
conda --version
```

#### 步骤4：创建一个新的 Python 环境（最关键）

无论 Windows / Linux，下面步骤完全一致。

创建一个医学影像项目专用环境，例如 `medimg`：

```bash
conda create -n medimg python=3.10
```

> 解释：
>
> - ​`-n medimg`：环境名字
> - ​`python=3.10`：指定 Python 版本（最兼容 MONAI/ASTRA/BART）

激活环境：

```bash
conda activate medimg
```

验证：

```bash
python --version
```

你应该看到：

```
Python 3.10.x
```

## 4.安装 ASTRA

### 4.1.ASTRA Toolbox 简介

> **ASTRA Toolbox 官方链接：**
>
> - 官方网站：https://astra-toolbox.com/
> - GitHub 仓库：https://github.com/astra-toolbox/astra-toolbox

​**ASTRA Toolbox**​（All Scale Tomographic Reconstruction Antwerp）是一个专门用于 **X 射线断层成像（CT）**  的高性能计算库。它提供了 GPU 加速的投影与重建算法，是科研人员进行：

- 平行束 / 扇束 / 圆锥束 CT 几何模拟
- 前向投影（forward projection）
- FBP（Filtered Back Projection）快速重建
- ART、SIRT、CGLS 等迭代重建算法
- 深度学习 + CT 模拟 / 重建前处理

的首选工具之一。

### 4.2.Windows 系统安装 ASTRA Toolbox

#### 通过 Conda 安装（最推荐）

这是针对 Windows 的​**首选方式**：简单、稳定、无须编译。

#### **步骤 1：创建 Python 环境**

建议先创建一个干净的 Conda 环境（避免污染系统 Python）：

```bash
conda create -n medimg python=3.10
conda activate medimg
```

> 推荐 Python 3.9–3.11（ASTRA 官方支持范围）。

#### **步骤 2：安装 ASTRA（CPU 版本）**

Windows 不支持 GPU 版本，因此直接用：

```bash
conda install -c astra-toolbox astra-toolbox
```

Conda 会自动安装：

- astra-toolbox（Python 接口）
- astra-core（C++ 核心库）
- 相关依赖（如 numpy、scipy）

### **步骤 3：验证安装**

进入 Python：

```bash
python
```

输入：

```python
import astra
print(astra.__version__)
```

正常输出版本号（如 `1.10.0`）即安装成功。

### 4.3.Linux 系统安装 ASTRA Toolbox

#### **步骤 1：创建环境（强烈推荐独立环境）**

```bash
conda create -n medimg python=3.10
conda activate medimg
```

#### **步骤 2：安装 ASTRA（自动选择 CPU/GPU）**

执行官方推荐命令：

```bash
conda install -c astra-toolbox -c nvidia astra-toolbox
```

说明：

- ​`-c astra-toolbox`：ASTRA 官方仓库
- ​`-c nvidia`：用于提供 CUDA runtime（Linux GPU 所需）
- 若你系统没有 GPU，会安装 CPU 版本
- 若你有 GPU，会安装 GPU 加速版（CUDA runtime 自动安装）

**不需要手动安装 CUDA toolkit！**

#### **步骤 3：测试安装**

```bash
python - << 'EOF'
import astra
print("ASTRA version:", astra.__version__)
EOF
```

输出版本号即成功。

### 4.4.测试安装结果

可以用此代码验证 ASTRA 是否能正常工作：

```python
import astra
import numpy as np

# 64x64 phantom
vol = np.ones((64, 64), dtype=np.float32)
vol_geom = astra.create_vol_geom(64, 64)

# 平行束几何（180 angles）
angles = np.linspace(0, np.pi, 180, endpoint=False)
proj_geom = astra.create_proj_geom('parallel', 1.0, 64, angles)

pid, sino = astra.create_sino(vol, proj_geom)
print("sino shape:", sino.shape)
```

预期输出：

```
sino shape: (180, 64)
```

## 5.安装 BART

### 5.1.BART 简介

> 官方网站：https://mrirecon.github.io/bart/  
> GitHub 仓库：https://github.com/mrirecon/bart  
> 官方文档：https://bart-doc.readthedocs.io/en/latest/intro.html  
> Workshop 与示例材料库：https://github.com/mrirecon/bart-workshop

**BART（Berkeley Advanced Reconstruction Toolbox）**  是一个用于 **MRI（磁共振成像）重建、信号处理和快速原型开发的开源工具箱**。它由 UC Berkeley 研发，属于医学影像和 MR 物理学界使用最广泛的重建工具之一，特别适合科研人员使用。

BART 是一个 ​**高性能 MRI 重建与信号处理工具包**，特点是：

- **命令行工具 + C 库 + Python/MATLAB 接口**
- 支持 ​**基本 MRI 重建**：

  - FFT / IFFT
  - Sense / pSense
  - GRAPPA
- 支持 ​**先进算法**：

  - CS（Compressed Sensing 压缩感知）
  - L1-wavelet / TV 正则
  - LLR, LORAKS, low-rank reconstruction
  - NUFFT（非均匀 FFT）
- 支持 MRI 采集几何：

  - 多通道（multi-coil）数据
  - k-space 非均匀采样
  - 多维数据 (2D/3D/dynamic MRI)

### 5.2.Windows 系统下安装 BART

#### 步骤1：环境准备

- Windows 系统。
- 安装一个兼容的 *GLC 编译器环境*（BART 在 Windows 上官方支持有限）。官方给出两种路径：使用 Cygwin 或者用虚拟机运行 Linux。
- 在 Cygwin 环境里，需要安装以下包：

  - Devel: gcc, make
  - Math: fftw3, fftw3-doc, libfftw3-devel, libfftw3\_3
  - Math: liblapack-devel, liblapack-doc, liblapack0
- 若希望 GPU 加速（Windows 本身支持较弱）—通常建议在 Linux 下做。文档中指出 “Running BART on Windows is *not supported*” 但有用户通过 Cygwin/WSL 运行。

#### 步骤2：下载 BART 源码或发行包

- 打开仓库： https://github.com/mrirecon/bart
- 或官方网页： https://mrirecon.github.io/bart/installation.html → 下载最新版 zip/tar 包。
- 下载最新 Release（如 version 0.9.00）中的 Windows 支持说明。
- 解压压缩文件到某个目录，例如 C:\\tools\\bart-0.9.00\\

#### 步骤3：使用 Cygwin 编译安装

- 安装 Cygwin：访问 https://www.cygwin.com/，下载安装程序。
- 在安装 Cygwin 时，在安装界面选择对应包（见 “环境准备”）。
- 打开 Cygwin Terminal，在解压的 BART 目录下运行：

  ```bash
  cd /cygdrive/c/tools/bart-0.9.00
  make
  ```

  这样会编译 BART 的命令行工具与库。
- 如果希望支持 GPU（视 Windows 下是否有适配、可能失败）可尝试在 Makefile 中添加 `CUDA=1`，但官方警告 Windows 支持有限。

#### 步骤4：在 Windows（Cygwin）中验证安装

1. 在 Cygwin shell 中，尝试运行某个 BART 工具命令：

    ```bash
    bart fft   # 如果此命令输出帮助或错误提示 “usage”, 则安装成功
    ```
2. 在 Python / MATLAB 中调用（如果你编译了 Python 接口）时，确保库路径已在环境变量中。
3. 若遇 “command not found” 或库加载失败，可检查环境变量 `PATH` 是否包含 BART 的 bin 目录，或者 Cygwin 的 usr/bin 是否已链接。

#### 提示与常见问题

- Windows 环境下 BART 支持程度低，**强烈推荐**使用 WSL2 或 Linux 虚拟机来运行更稳定。文档明确指出 “Windows support by MSYS2; generic” 或 “Use Cygwin” 等语句。
- 若只是为了算法复现实验，也可以在 Windows 上启用 WSL2 安装 Ubuntu，然后在 WSL2 内按照 Linux 安装流程运行。
- 编译过程中常见错误：缺少 `fftw3`​、`lapack`​、`blas` 库，确保在 Cygwin 安装时选择对应 dev 包。
- GPU 支持在 Windows 极为不稳定，不推荐用于研究初期。

### 5.3.Linux 系统下安装 BART

### 步骤1：环境准备

推荐先安装的依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
    gcc \
    make \
    libfftw3-dev \
    liblapacke-dev \
    libpng-dev \
    libopenblas-dev
```

- ​`gcc`​, `make`：编译工具。
- ​`libfftw3-dev`：FFTW（快速傅里叶变换库）开发包。
- ​`liblapacke-dev`：LAPACK/BLAS 相关开发包。
- ​`libpng-dev`：若需要图像读取/写出支持。
- ​`libopenblas-dev`：推荐加速 BLAS 运算。

#### 步骤2：下载 BART 源码或 Release 包

往官方页面或 GitHub 仓库下载最新版本：

```bash
git clone https://github.com/mrirecon/bart.git
cd bart
```

或者从 release 页面下载 `.tar.gz`：

```bash
wget https://github.com/mrirecon/bart/archive/v0.9.00.tar.gz
tar xzvf v0.9.00.tar.gz
cd bart-0.9.00
```

### 步骤3：编译安装、开启 GPU 加速（可选）

在源码目录中运行：

```bash
make
```

如果你想启用额外功能（如 ISMRMRD 支持）：

```bash
make ismrmrd
```

如果你的机器有 NVIDIA 显卡并且你希望使用 GPU 加速，BART 支持 CUDA。

在编译时在源码目录执行：

```bash
make clean
make CUDA=1
```

即可启用 GPU 支持。需要安装对应的 CUDA 版本、NVIDIA 驱动、CUDA Toolkit 等。

### 步骤4：安装 Python 接口（可选）

如果你打算在 Python 中使用 BART，很多版本中源码自带 `python/` 目录。建议：

```bash
cd python
pip install .
```

然后在 Python 中测试：

```python
import bart
bart.print_settings()
```

#### 步骤5：验证安装是否成功

在终端执行：

```bash
bart fft   # 显示帮助说明
```

也可以运行一个简单命令行重建工具测试。若能输出版本号、帮助信息或不报错，即表示安装成功。

## 6.安装 MONAI

### 6.1.MONAI 简介

> - 官方网站：https://monai.dev/
> - GitHub 仓库：https://github.com/Project-MONAI/MONAI
> - 官方文档：https://docs.monai.io/（包括安装指导、API 文档等）

**MONAI（Medical Open Network for AI）**  是由 **NVIDIA + 美国 NIH（国立卫生研究院）**  共同主导开发的 ​**医学影像深度学习框架**。  
它基于 PyTorch，专门为医学影像任务（CT / MRI / X-ray / Ultrasound）设计，提供从数据加载、预处理、训练、评估到部署的完整工具链。

MONAI 已成为医学影像 AI 研究中最主流的开源框架之一。

### 6.2.安装前准备

在安装 MONAI 之前，需要：

- 安装 **Miniconda 或 Anaconda**
- 安装 **Python 3.9–3.11（推荐 3.10）**
- 可选：安装 NVIDIA 驱动（如需 GPU）
- 推荐使用 **独立 conda 环境**

### 6.3.安装 PyTorch

MONAI 依赖 PyTorch，因此必须先安装 PyTorch。

前往 PyTorch 官方安装页面：

👉 [https://pytorch.org/](https://pytorch.org/)

选择对应系统的指令。

● CPU 版本（Windows / Linux 通用）

```bash
pip install torch torchvision torchaudio
```

● GPU 版本（以 CUDA 12.4 为例）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> 如果你不知道 CUDA 版本，运行 `nvidia-smi`（Linux）或查看 NVIDIA 控制面板（Windows）。

### 6.4.安装 MONAI

```bash
pip install monai
```

**安装可选扩展依赖（可选）**

常用增强包：

```bash
pip install "monai[nibabel,skimage,pillow,ignite]"
```

全部可选依赖（最完整版）：

```bash
pip install "monai[all]"
```

适用于科研任务（如 segmentation / registration）。

### 6.5.安装验证

运行以下 Python 脚本：

```python
import monai
import torch

print("MONAI version:", monai.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

若无报错，且输出现有版本，则说明安装成功。

## 7.安装 NiBabel

### 7.1.NiBabel 简介

> - GitHub 仓库（源代码 + issues +开发者） : https://github.com/nipy/nibabel
> - 官方文档 : https://nipy.org/nibabel/
> - PyPI : https://pypi.org/project/nibabel/

NiBabel 是一个专门用于读取、处理和保存医学影像文件的 Python 工具包，支持 NIfTI（.nii/.nii.gz）、Analyze、MINC、MGH/MGZ 等主流格式，并能将医学影像数据加载为 NumPy 数组，同时提供 affine 矩阵、空间方向、头信息等完整元数据管理。它是医学影像 AI、神经影像（fMRI/dMRI）、深度学习预处理与科研分析中最常用的文件 I/O 库，也是 MONAI、PyTorch 及各种医学影像工具链的底层基础组件之一。

### 7.2.安装当前稳定版（推荐）

使用 pip 安装 NiBabel 的最新发布版本：

```bash
pip install nibabel
```

### 7.3 安装最新开发版

如果希望使用 GitHub 上的最新开发进度（未发布版本），可运行：

```bash
pip install git+https://github.com/nipy/nibabel
```

### 7.4 以"可编辑模式"安装（开发 NiBabel 源码时使用）

当你需要修改 NiBabel 源码、参与开发或调试时，可以以可编辑模式安装：

```bash
git clone https://github.com/nipy/nibabel.git
pip install -e ./nibabel
```

这种方式会让 Python 直接引用本地源码目录，修改后无需重新安装。

### 7.5.测试 NiBabel

#### **1. 在开发过程中运行完整测试（推荐开发者使用 tox）**

```bash
git clone https://github.com/nipy/nibabel.git
cd nibabel
tox
```

​`tox` 会自动创建虚拟环境并在多个 Python 版本下测试 NiBabel，适合参与开发的用户。

#### **2. 测试已安装的 NiBabel**

如果你只想测试当前系统中安装的 NiBabel，可安装测试依赖并运行 pytest：

```bash
pip install nibabel[test]
pytest --pyargs nibabel
```

这将运行 NiBabel 的全部测试模块，确保安装正常。

‍
