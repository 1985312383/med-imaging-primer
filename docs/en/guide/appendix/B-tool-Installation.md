---
title: Appendix B Tool Installation
description: Introduces installation methods for common tools.
---

# Appendix B: Tool Installation

# 1. Introduction

Medical imaging algorithm development typically relies on the collaboration of multiple toolchains. This chapter will introduce the core environments required for this tutorial, including **Python, ASTRA, BART, MONAI, NiBabel** and other basic components. We will start with basic Python environment configuration and gradually explain how to install CT reconstruction library (ASTRA), MRI reconstruction toolbox (BART), deep learning framework (MONAI), and medical imaging file reading/writing tools (NiBabel). Through this section, readers can quickly build a complete, reproducible medical imaging development and reconstruction experimental environment.

# 2. Direct Python Installation

## 2.1. Windows System

### 2.1.1. Download Python Installer

**Recommended method: Download the installer from the official Python website**

1. Open your browser and visit: `https://www.python.org`
2. Click **Downloads** in the top navigation bar
3. The page will automatically recommend the version suitable for Windows, for example: **Download Python 3.xx**
4. Click this button to download a `.exe` installer (e.g., `python-3.12.3-amd64.exe`)

### 2.1.2. Run the Installer (Most Critical Step)

Double-click the downloaded `python-3.xx-amd64.exe` and follow these steps:

1. At the bottom of the installation interface, there is a line:

    - ✅ **Must check:** **`Add Python 3.xx to PATH`**
    - This step determines whether you can directly use the `python` command in the command line later, which is very important.
2. In **Customize installation**:

    1. **Optional Features**
        It is recommended to check all:

        - ✅ `pip` (Python package manager, essential)
        - ✅ `IDLE` (built-in small editor for visual debugging of simple scripts)
        - ✅ `Documentation`
        - ✅ `Python test suite`
        - ✅ `py launcher`
        Then click **Next**
    2. **Advanced Options**:
        It is recommended to check:

        - ✅ `Install for all users` (recommended for multi-user computers, path will be in `C:\Program Files\Python3x`)
        - ✅ `Add Python to environment variables` (if you missed checking PATH earlier, you must check it here)
        - ✅ `Precompile standard library` (speeds up first run)

### 2.1.3. Verify Python Installation

After installation is complete, follow these steps to check:

1. Press the `Win` key, type `cmd`, and open **Command Prompt**
2. In the black window, type:

    ```bash
    python --version
    ```

    Or

    ```bash
    py --version
    ```
3. If you see something like:

    ```text
    Python 3.12.3
    ```

    It means the installation was successful and PATH is working.

---

### 2.1.4. Verify pip Availability

In the same command line, type:

```bash
pip --version
```

If you see something like:

```text
pip 24.x from C:\Python\Python312\Lib\site-packages\pip (python 3.12)
```

It means `pip` has been successfully installed and can be used to install subsequent packages like `monai` and `nibabel`.

## 2.2. Linux System

### 2.2.1. Check if Python is Already Installed

Open the terminal and type:

```bash
python3 --version
```

If you see output similar to:

```text
Python 3.10.12
```

It means the system already has Python3, which can generally be used directly.

Then check if `pip` exists:

```bash
pip3 --version
```

If it prompts "command not found", it means pip is not installed yet and will be installed later.

### 2.2.2. Install Python Using Package Manager (Ubuntu / Debian)

#### 2.1 Update Software Sources

First update the package list:

```bash
sudo apt update
```

#### 2.2 Install Python3 and Common Dependencies

Execute:

```bash
sudo apt install -y python3 python3-pip python3-venv
```

Description:

- `python3`: Python interpreter
- `python3-pip`: Python package management tool pip
- `python3-venv`: Used to create virtual environments (very useful when not installing Conda)

After installation, confirm the versions again:

```bash
python3 --version
pip3 --version
```

# 3. Install Python Using Conda (Recommended)

Conda is the most commonly used environment management tool in the field of data science and medical imaging, capable of creating independent Python environments for different projects to avoid package conflicts.  
This section will introduce how to install Conda (Miniconda) on **Windows / Linux** and create a clean, controllable Python environment.

## 3.1. Download and Install Miniconda (Recommended)

**Why use Miniconda instead of Anaconda?**

- Miniconda is more lightweight and doesn't install thousands of unnecessary packages
- More suitable for scientific research projects and medical imaging engineering
- Fully compatible with Anaconda's functionality

## 3.2. Windows Miniconda Installation

### **Step 1: Download Installer**

Visit the official website:

👉 https://repo.anaconda.com/miniconda/

Select:

- **Miniconda3 Windows 64-bit Installer (.exe)**

### **Step 2: Run Installer**

Double-click the installer:

1. Choose **Just Me** or **All Users**
2. **Important: Check "Add Miniconda3 to my PATH environment variable"**

    - If you don't check this, it's okay, Conda will automatically add PATH for CMD / PowerShell
3. Click Install

### **Step 3: Verify Installation**

Open *Anaconda Prompt* or CMD:

```bash
conda --version
```

If it displays:

```
conda 24.x.x
```

Then the installation is successful.

## 3. Linux Miniconda Installation

### **Step 1: Download Installation Script**

Visit the official website:

👉 [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

Copy the Linux installation script link, for example:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### **Step 2: Run Installation Script**

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts:

- Read and agree to the license
- Choose installation path (default is fine: `~/miniconda3`)
- Choose whether to initialize conda (recommended YES)

### **Step 3: Activate Conda**

If you chose YES earlier, just reopen the terminal.  
If you chose NO, you need to manually run:

```bash
source ~/miniconda3/bin/activate
```

Verify:

```bash
conda --version
```

### Step 4: Create a New Python Environment (Most Critical)

Whether on Windows / Linux, the following steps are completely consistent.

Create a dedicated environment for medical imaging projects, for example `medimg`:

```bash
conda create -n medimg python=3.10
```

> Explanation:
>
> - `-n medimg`: Environment name
> - `python=3.10`: Specify Python version (most compatible with MONAI/ASTRA/BART)

Activate the environment:

```bash
conda activate medimg
```

Verify:

```bash
python --version
```

You should see:

```
Python 3.10.x
```

# 4. Install ASTRA

## 4.1. ASTRA Toolbox Introduction

> **ASTRA Toolbox Official Links:**
>
> - Official website: https://astra-toolbox.com/
> - GitHub repository: https://github.com/astra-toolbox/astra-toolbox

**ASTRA Toolbox** (All Scale Tomographic Reconstruction Antwerp) is a high-performance computing library specifically designed for **X-ray tomography (CT)**. It provides GPU-accelerated projection and reconstruction algorithms, making it one of the preferred tools for researchers to perform:

- Parallel beam / fan beam / cone beam CT geometric simulation
- Forward projection
- FBP (Filtered Back Projection) fast reconstruction
- Iterative reconstruction algorithms such as ART, SIRT, CGLS
- Deep learning + CT simulation / reconstruction preprocessing

## 4.2. Windows System ASTRA Toolbox Installation

### Install via Conda (Most Recommended)

This is the **preferred method** for Windows: simple, stable, and no compilation required.

### **Step 1: Create Python Environment**

It is recommended to first create a clean Conda environment (to avoid contaminating the system Python):

```bash
conda create -n medimg python=3.10
conda activate medimg
```

> Python 3.9–3.11 is recommended (ASTRA official support range).

### **Step 2: Install ASTRA (CPU Version)**

Windows does not support GPU version, so use directly:

```bash
conda install -c astra-toolbox astra-toolbox
```

Conda will automatically install:

- astra-toolbox (Python interface)
- astra-core (C++ core library)
- Related dependencies (such as numpy, scipy)

### **Step 3: Verify Installation**

Enter Python:

```bash
python
```

Type:

```python
import astra
print(astra.__version__)
```

If the version number (such as `1.10.0`) is output normally, the installation is successful.

## 4.2. Linux System ASTRA Toolbox Installation

### **Step 1: Create Environment (Strongly Recommended Independent Environment)**

```bash
conda create -n medimg python=3.10
conda activate medimg
```

### **Step 2: Install ASTRA (Automatically Select CPU/GPU)**

Execute the officially recommended command:

```bash
conda install -c astra-toolbox -c nvidia astra-toolbox
```

Description:

- `-c astra-toolbox`: ASTRA official repository
- `-c nvidia`: Used to provide CUDA runtime (required for Linux GPU)
- If your system does not have a GPU, the CPU version will be installed
- If you have a GPU, the GPU-accelerated version will be installed (CUDA runtime automatically installed)

**No need to manually install CUDA toolkit!**

### **Step 3: Test Installation**

```bash
python - << 'EOF'
import astra
print("ASTRA version:", astra.__version__)
EOF
```

If the version number is output, it's successful.

## 4.3. Test Installation Results

You can use this code to verify if ASTRA is working properly:

```python
import astra
import numpy as np

# 64x64 phantom
vol = np.ones((64, 64), dtype=np.float32)
vol_geom = astra.create_vol_geom(64, 64)

# Parallel beam geometry (180 angles)
angles = np.linspace(0, np.pi, 180, endpoint=False)
proj_geom = astra.create_proj_geom('parallel', 1.0, 64, angles)

pid, sino = astra.create_sino(vol, proj_geom)
print("sino shape:", sino.shape)
```

Expected output:

```
sino shape: (180, 64)
```

# 5. Install BART

## 5.1. BART Introduction

> Official website: https://mrirecon.github.io/bart/  
> GitHub repository: https://github.com/mrirecon/bart  
> Official documentation: https://bart-doc.readthedocs.io/en/latest/intro.html  
> Workshop and example material repository: https://github.com/mrirecon/bart-workshop

**BART (Berkeley Advanced Reconstruction Toolbox)** is an open-source toolbox for **MRI (Magnetic Resonance Imaging) reconstruction, signal processing, and rapid prototyping development**. It was developed by UC Berkeley and is one of the most widely used reconstruction tools in the medical imaging and MR physics community, especially suitable for researchers.

BART is a **high-performance MRI reconstruction and signal processing toolkit** characterized by:

- **Command-line tools + C library + Python/MATLAB interfaces**
- Supports **basic MRI reconstruction**:

  - FFT / IFFT
  - Sense / pSense
  - GRAPPA
- Supports **advanced algorithms**:

  - CS (Compressed Sensing)
  - L1-wavelet / TV regularization
  - LLR, LORAKS, low-rank reconstruction
  - NUFFT (Non-uniform FFT)
- Supports MRI acquisition geometries:

  - Multi-coil data
  - k-space non-uniform sampling
  - Multi-dimensional data (2D/3D/dynamic MRI)

## 5.2. Windows System BART Installation

### Step 1: Environment Preparation

- Windows system.
- Install a compatible *GLC compiler environment* (BART has limited official support on Windows). The official provides two paths: use Cygwin or run Linux in a virtual machine.
- In the Cygwin environment, you need to install the following packages:

  - Devel: gcc, make
  - Math: fftw3, fftw3-doc, libfftw3-devel, libfftw3\_3
  - Math: liblapack-devel, liblapack-doc, liblapack0
- If you want GPU acceleration (Windows itself has weak support)—it's generally recommended to do this on Linux. The documentation points out that "Running BART on Windows is *not supported*" but some users run it through Cygwin/WSL.

### Step 2: Download BART Source Code or Release Package

- Open the repository: https://github.com/mrirecon/bart
- Or the official webpage: https://mrirecon.github.io/bart/installation.html → Download the latest version zip/tar package.
- Download the latest Release (such as version 0.9.00) with Windows support instructions.
- Extract the compressed file to a directory, for example C:\\tools\\bart-0.9.00\\

### Step 3: Compile and Install Using Cygwin

- Install Cygwin: Visit https://www.cygwin.com/, download the installer.
- When installing Cygwin, select the corresponding packages in the installation interface (see "Environment Preparation").
- Open Cygwin Terminal, in the extracted BART directory, run:

  ```bash
  cd /cygdrive/c/tools/bart-0.9.00
  make
  ```

  This will compile BART's command-line tools and libraries.
- If you want to support GPU (depends on whether there's an adaptation on Windows, it may fail), you can try adding `CUDA=1` in the Makefile, but the official warns that Windows support is limited.

### Step 4: Verify Installation on Windows (Cygwin)

1. In the Cygwin shell, try running a BART tool command:

    ```bash
    bart fft   # If this command outputs help or an error prompt "usage", then installation is successful
    ```
2. When calling in Python / MATLAB (if you compiled the Python interface), ensure the library path is in the environment variables.
3. If you encounter "command not found" or library loading failure, you can check if the environment variable `PATH` contains BART's bin directory, or if Cygwin's usr/bin has been linked.

### Tips and Common Issues

- BART support is low in Windows environments, **strongly recommend** using WSL2 or Linux virtual machines for more stable operation. The documentation explicitly states statements like "Windows support by MSYS2; generic" or "Use Cygwin".
- If just for algorithm reproduction experiments, you can also enable WSL2 on Windows to install Ubuntu, then follow the Linux installation process within WSL2.
- Common errors during compilation: missing `fftw3`, `lapack`, `blas` libraries, ensure you select the corresponding dev packages when installing Cygwin.
- GPU support is extremely unstable on Windows and not recommended for early research stages.

## 5.3. Linux System BART Installation

### Step 1: Environment Preparation

Recommended dependencies to install first:

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

- `gcc`, `make`: Compilation tools.
- `libfftw3-dev`: FFTW (Fast Fourier Transform Library) development package.
- `liblapacke-dev`: LAPACK/BLAS related development package.
- `libpng-dev`: If you need image reading/writing support.
- `libopenblas-dev`: Recommended to accelerate BLAS operations.

### Step 2: Download BART Source Code or Release Package

Download the latest version from the official page or GitHub repository:

```bash
git clone https://github.com/mrirecon/bart.git
cd bart
```

Or download `.tar.gz` from the release page:

```bash
wget https://github.com/mrirecon/bart/archive/v0.9.00.tar.gz
tar xzvf v0.9.00.tar.gz
cd bart-0.9.00
```

### Step 3: Compile Install, Enable GPU Acceleration (Optional)

In the source code directory, run:

```bash
make
```

If you want to enable additional features (such as ISMRMRD support):

```bash
make ismrmrd
```

If your machine has an NVIDIA graphics card and you want to use GPU acceleration, BART supports CUDA.

Execute in the source code directory during compilation:

```bash
make clean
make CUDA=1
```

This enables GPU support. You need to install the corresponding CUDA version, NVIDIA driver, CUDA Toolkit, etc.

### Step 4: Install Python Interface (Optional)

If you plan to use BART in Python, many versions have a built-in `python/` directory in the source code. It is recommended:

```bash
cd python
pip install .
```

Then test in Python:

```python
import bart
bart.print_settings()
```

### Step 5: Verify Installation is Successful

Execute in the terminal:

```bash
bart fft   # Display help information
```

You can also run a simple command-line reconstruction tool to test. If it can output the version number, help information, or no errors, it indicates successful installation.

# 6. Install MONAI

## 6.1. MONAI Introduction

> - Official website: https://monai.dev/
> - GitHub repository: https://github.com/Project-MONAI/MONAI
> - Official documentation: https://docs.monai.io/ (including installation guide, API documentation, etc.)

**MONAI (Medical Open Network for AI)** is a **medical imaging deep learning framework** jointly led by **NVIDIA + US NIH (National Institutes of Health)**.  
It is based on PyTorch and specifically designed for medical imaging tasks (CT / MRI / X-ray / Ultrasound), providing a complete toolchain from data loading, preprocessing, training, evaluation to deployment.

MONAI has become one of the most mainstream open-source frameworks in medical imaging AI research.

## 6.2. Pre-Installation Preparation

Before installing MONAI, you need to:

- Install **Miniconda or Anaconda**
- Install **Python 3.9–3.11 (3.10 recommended)**
- Optional: Install NVIDIA drivers (if GPU is needed)
- It is recommended to use an **independent conda environment**

## 6.3. Install PyTorch

MONAI depends on PyTorch, so PyTorch must be installed first.

Go to the PyTorch official installation page:

👉 [https://pytorch.org/](https://pytorch.org/)

Select the corresponding system's command.

● CPU Version (Windows / Linux Universal)

```bash
pip install torch torchvision torchaudio
```

● GPU Version (using CUDA 12.4 as an example)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> If you don't know the CUDA version, run `nvidia-smi` (Linux) or check the NVIDIA Control Panel (Windows).

## 6.4. Install MONAI

```bash
pip install monai
```

**Install Optional Extension Dependencies (Optional)**

Common enhancement packages:

```bash
pip install "monai[nibabel,skimage,pillow,ignite]"
```

All optional dependencies (most complete version):

```bash
pip install "monai[all]"
```

Suitable for research tasks (such as segmentation / registration).

## 6.5. Installation Verification

Run the following Python script:

```python
import monai
import torch

print("MONAI version:", monai.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

If there are no errors and the existing versions are output, it means the installation is successful.

# 7. Install NiBabel

## 7.1. NiBabel Introduction

> - GitHub repository (source code + issues + developers): https://github.com/nipy/nibabel
> - Official documentation: https://nipy.org/nibabel/
> - PyPI: https://pypi.org/project/nibabel/

NiBabel is a Python toolkit specifically designed for reading, processing, and saving medical imaging files, supporting mainstream formats such as NIfTI (.nii/.nii.gz), Analyze, MINC, MGH/MGZ, etc. It can load medical imaging data as NumPy arrays while providing complete metadata management such as affine matrices, spatial orientations, and header information. It is one of the most commonly used file I/O libraries in medical imaging AI, neuroimaging (fMRI/dMRI), deep learning preprocessing, and scientific analysis, and is also one of the underlying basic components of MONAI, PyTorch, and various medical imaging toolchains.

## 7.2. **Install Current Stable Version (Recommended)**

Use pip to install the latest released version of NiBabel:

```bash
pip install nibabel
```

## 7.3. **Install Latest Development Version**

If you want to use the latest development progress on GitHub (unreleased version), you can run:

```bash
pip install git+https://github.com/nipy/nibabel
```

## 7.4. **Install in "Editable Mode" (Used when Developing NiBabel Source Code)**

When you need to modify NiBabel source code, participate in development, or debug, you can install in editable mode:

```bash
git clone https://github.com/nipy/nibabel.git
pip install -e ./nibabel
```

This method makes Python directly reference the local source code directory, so no reinstallation is needed after modifications.

## 7.5. Test NiBabel

### **1. Run Complete Tests During Development (Recommended for Developers Using tox)**

```bash
git clone https://github.com/nipy/nibabel.git
cd nibabel
tox
```

`tox` will automatically create virtual environments and test NiBabel under multiple Python versions, suitable for users participating in development.

### **2. Test Installed NiBabel**

If you just want to test the NiBabel installed in your current system, you can install test dependencies and run pytest:

```bash
pip install nibabel[test]
pytest --pyargs nibabel
```

This will run all test modules of NiBabel to ensure the installation is normal.