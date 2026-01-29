---
title: 附录A：关键公式
description: 汇总医学影像中最常见、最重要、跨模态通用的公式。使读者在阅读主教程时能快速查阅与理解数学推导。适合零基础读者快速建立公式框架。
---

# 附录A：关键公式

## 1.引言

医学影像是跨越物理学、数学、工程学与医学的典型交叉领域。无论是 CT 的投影重建、MRI 的傅里叶编码，还是 PET 的统计建模，这些成像技术的核心都建立在一系列清晰、可解释、可验证的数学公式之上。理解这些公式不仅能帮助读者更好地掌握影像生成的原理，也能为深入学习图像重建、图像处理、深度学习建模等后续内容奠定坚实基础。

本章旨在汇总医学影像中最常用、最关键、跨模态通用的数学公式，并按照“从基础概念 → 各模态成像模型 → 图像处理 → 深度学习”的路径进行组织。读者可以将本章视作一个随查随用的快速参考（Quick Reference），在学习主教程时随时回溯；也可以作为进一步研究影像物理与重建算法时的入门索引。

为了保证连贯性，本章采用简明的符号体系与统一的表述方式，并在必要时提供物理意义的解释。即便没有深厚的数学基础，读者也能通过本章建立对医学影像数学框架的理解，并将其应用于实践中的图像分析与算法开发。

## 2.医学影像通用数学基础

本节介绍医学影像中跨模态通用的数学原理，包括坐标系、采样、卷积及噪声建模。这些概念在 CT、MRI、X-ray、PET、超声的建模与重建中均扮演关键角色。

### 2.1 坐标系与图像几何

医学影像中的几何关系主要涉及 **像素/体素坐标 → 物理世界坐标** 的映射，以及常见的仿射变换(旋转、缩放、平移等)。

 **(1)像素/体素坐标 → 物理空间坐标**

影像文件(如 DICOM、NIfTI)通常提供一个 4×4 的空间变换矩阵，用于将体素坐标 $(i, j, k)$ 转换为物理坐标 $(x, y, z)$：

$$
\left[\begin{array}{c}
x \\
y \\
z \\
1
\end{array}\right]=
\mathbf{M}_{\text {DICOM }}\left[\begin{array}{c}
i \\
j \\
k \\
1
\end{array}\right]
$$

- $\mathbf{M}_{\text{DICOM}}$ 包含：

  - 体素间距(spacing)
  - 图像方向(orientation)
  - 图像原点(origin)

---

 **(2)DICOM 空间变换矩阵（典型形式）**

$$
\mathbf{M}_{\text{DICOM}}=
\left[\begin{array}{cc}
\mathbf{R} \cdot \text{diag}(\Delta x,\Delta y,\Delta z) & \mathbf{T} \\
0 & 1
\end{array}\right]
$$

其中：

- $\mathbf{R}$：方向余弦矩阵(3×3，指定图像轴在世界坐标系的方向)
- $\Delta x, \Delta y, \Delta z$：体素大小(单位 mm)
- $\mathbf{T}$：原点坐标(患者坐标系下的 mm)

---

 **(3)常见几何变换矩阵**

● 缩放(Scaling)

$$
\mathbf{S} =
\left[\begin{array}{ccc}
s_x & 0 & 0 \\
0 & s_y & 0 \\
0 & 0 & s_z
\end{array}\right]
$$

● 旋转(Rotation，以 z 轴为例)

$$
\mathbf{R}_z(\theta)=
\left[\begin{array}{ccc}
\cos\theta & -\sin\theta & 0\\
\sin\theta & \cos\theta  & 0\\
0 & 0 & 1
\end{array}\right]
$$

● 仿射变换(Affine Transform)

$$
\mathbf{x}' = \mathbf{A}\mathbf{x} + \mathbf{b}
$$

在齐次坐标中表达为：

$$
\left[\begin{array}{c}
\mathbf{x}' \\
1
\end{array}\right]=
\left[\begin{array}{cc}
\mathbf{A} & \mathbf{b}\\
0 & 1
\end{array}\right]
\left[\begin{array}{c}
\mathbf{x} \\
1
\end{array}\right]
$$

仿射变换广泛用于图像配准、重采样和多模态对齐。

### 2.2 图像采样与卷积

---

 **(1)卷积定义（连续/离散）**

● 连续卷积

$$
(f * g)(t)=\int_{-\infty}^{+\infty} f(\tau)\, g(t-\tau)\, d\tau
$$

● 离散卷积(信号)

$$
(f * g)[n] = \sum_{k=-\infty}^{+\infty} f[k]\, g[n-k]
$$

● 2D 图像卷积(常用于滤波)

$$
I'(x,y)=\sum_m \sum_n I(x-m,y-n) \, K(m,n)
$$

卷积在 CT 滤波反投影(FBP)、MRI 去噪、图像平滑、锐化中都被频繁使用。

---

 **(2)下采样与上采样**

● 下采样(Downsampling)

$$
x_{\text{down}}[n] = x[kN]
$$

N \= 2 时表示宽高减半。

● 上采样(Upsampling)

插 0 然后滤波：

$$
x_{\text{up}}[n] =
\begin{cases}
x[n/N], & n \mod N = 0 \\
0, & \text{otherwise}
\end{cases}
$$

---

 **(3)常见插值公式**

● 双线性插值(2D)

$$
f(x,y)=\sum_{m=0}^1\sum_{n=0}^1 
f(i+m, j+n)(1-|x-i-m|)(1-|y-j-n|)
$$

● 三线性插值(3D)

用于体数据(CT/MRI)重采样，为双线性插值的三维扩展。

---

 **(4)空域 ↔ 频域关系（傅里叶变换基础）**

● 傅里叶变换(连续)

$$
F(\omega)=\int f(t)e^{-j\omega t}dt
$$

● 离散傅里叶变换(DFT)

$$
X[k]=\sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}
$$

● 卷积定理(医学影像极为重要)

$$
\mathcal{F}\{f*g\} = \mathcal{F}\{f\}\cdot \mathcal{F}\{g\}
$$

说明：  
​**卷积在频域等价于乘法**，是 CT 滤波反投影(FBP)和 MRI 重建的核心。

### 2.3 噪声模型

医学影像中常见的噪声主要来自探测器、电子噪声、光子统计噪声等。

---

 **(1)加性高斯噪声(MRI / CT 常见)**

$$
y = x + \mathcal{N}(0,\sigma^2)
$$

适用场景：

- CT 重建后图像
- MRI 实部/虚部信号

---

 **(2)泊松噪声(PET / X-ray)**

PET、X-ray 属于 ​**光子计数过程**，天然满足泊松模型：

$$
y \sim \text{Poisson}(x)
$$

应用：

- PET 投影数据(sinogram)
- X-ray 强度采集
- CT 投影数据的噪声建模

泊松噪声在低剂量 CT、低计数 PET 中尤为显著。

## 3. CT（Computed Tomography）的关键公式

本节总结 CT 成像与重建中最核心的数学公式，包括 X 射线衰减模型、Radon 变换、滤波反投影(FBP)以及 CT 图像的 HU 转换和窗口化处理。这些公式构成了现代 CT 成像技术的数学基础。

---

### 3.1 X-ray 衰减模型

X 射线穿过组织时会被吸收与散射，其衰减过程由 **Lambert–Beer 定律** 描述，是 CT 物理模型的起点。

---

 **(1)Lambert–Beer 定律**

当 X 射线束穿过介质时，探测器接收到的强度 $I$ 与入射强度 $I_0$ 的关系为：

$$
I = I_0 \exp\left( -\int_L \mu(s)\, ds \right)
$$

其中：

- $\mu(s)$：线性衰减系数 (linear attenuation coefficient)
- $L$：光线路径
- $I_0$：入射光子强度
- $I$：探测器接收到的强度

---

 **(2)投影数据(log-transform)**

CT 投影通常进行对数变换，将指数衰减线性化：

$$
p = -\ln \left( \frac{I}{I_0} \right)
$$

可得到：

$$
p = \int_L \mu(s)\, ds
$$

这就是 Radon 变换的物理来源。

---

### 3.2 Radon 变换

Radon 变换描述了物体在不同角度下的线积分，是 CT 投影数据(sinogram)的数学表示。

---

 **(1)Radon 变换定义**

二位对象 $\mu(x,y)$ 在角度 $\theta$ 上的投影写为：

$$
p(\theta, t) = 
\int_{-\infty}^{+\infty}
\mu(x, y)\, ds
$$

将积分路径参数化后：

$$
p(\theta, t)=\int_{-\infty}^{+\infty} \mu(t \cos \theta-s \sin \theta, t \sin \theta+s \cos \theta) d s
$$

其中：

- $\theta$：射线方向
- $t$：射线到原点的垂直距离
- $p(\theta,t)$：sinogram 数据

---

 **(2)Radon 反变换(理想情况)**

理论上，物体可通过 Radon 反变换恢复：

$$
\mu(x,y)=\mathcal{R}^{-1}\{p(\theta,t)\}
$$

实际重建必须结合滤波反投影(FBP)。

---

### 3.3 滤波反投影（FBP）

FBP(Filtered Back Projection)是经典 CT 重建算法，是目前临床普遍使用的重建框架之一。

---

 **(1)卷积滤波(Filtering Step)**

对每个角度的投影 $p(\theta,t)$ 做滤波：

$$
\tilde{p}(\theta,t)=
p(\theta,t) * h(t)
$$

其中：

- $h(t)$：重建滤波器，例如

  - ​**Ram-Lak**(理想高通)
  - Shepp-Logan
  - Hanning、Cosine 等

Ram-Lak 滤波器的频域表达：

$$
H(\omega)=|\omega|
$$

滤波的作用是补偿反投影带来的低频增强。

---

 **(2)反投影(Back Projection)**

将所有角度的滤波投影反投影到图像空间：

$$
\mu(x, y)=\int_0^\pi \tilde{p}(\theta, x \cos \theta+y \sin \theta) d \theta
$$

意义：

- 将 1D 滤波投影“拉回”属于它的二维位置
- 所有角度累加即可恢复物体衰减系数分布

---

 **(3)FBP 总公式**

$$
\mu(x, y)=\int_0^\pi[p(\theta, t) * h(t)]_{t=x \cos \theta+y \sin \theta} d \theta
$$

---

### 3.4 CT 值（HU）转换公式

CT 图像通常以 HU(Hounsfield Unit)为单位，用来衡量组织对 X 射线的衰减能力。

---

 **(1)HU 标准化公式**

$$
\text{HU} = 1000 \cdot 
\frac{\mu - \mu_{\text{water}}}{\mu_{\text{water}}}
$$

- $\mu$：组织衰减系数
- $\mu_{\text{water}}$：水的衰减系数(基准)
- 水 \= 0 HU
- 空气 \= –1000 HU

---

 **(2)线性窗函数(Windowing)**

显示 CT 图像时需进行 windowing：

$$
I_{\text{display}}=
\text{clip}\left(
\frac{HU - (WL - \frac{WW}{2})}{WW}
,\, 0,\, 1
\right)
$$

其中：

- $WW$：Window Width(窗宽)
- $WL$：Window Level(窗位)
- clip(·)：将结果限制在 [0,1]

示例：

- 软组织窗口：WL\=40, WW\=400
- 肺窗口：WL\=-600, WW\=1500
- 骨窗口：WL\=400, WW\=2000

## 4. MRI（Magnetic Resonance Imaging）的关键公式

MRI(磁共振成像)的数学基础主要包括核磁共振信号的产生(Larmor 进动与 Bloch 方程)、组织弛豫过程(T1、T2)、以及 k-space 的傅里叶编码与重建。本节给出 MRI 成像中最核心的公式。

---

### 4.1 脉冲序列基础

MRI 信号由磁场与射频脉冲(RF pulse)作用下的核磁矩行为决定，基本动力学由 Larmor 频率和 Bloch 方程描述。

---

 **(1)Larmor 频率公式**

核自旋在静磁场 $B_0$ 中的进动角频率为：

$$
\omega_0 = \gamma B_0
$$

其中：

- $\omega_0$：Larmor 频率
- $\gamma$：旋磁比(proton: $\gamma/2\pi \approx 42.58\text{ MHz/T}$)
- $B_0$：主磁场强度

说明：高场强(如 3T)会带来更高的信噪比(SNR)。

---

 **(2)Bloch 方程(简式)**

描述磁化矢量 $\mathbf{M} = (M_x, M_y, M_z)$ 在磁场中的动态变化：

$$
\frac{d \mathbf{M}}{d t}=\gamma(\mathbf{M} \times \mathbf{B})-\frac{M_x \hat{i}+M_y \hat{j}}{T_2}-\frac{\left(M_z-M_0\right) \hat{k}}{T_1}
$$

其中：

- 第一项：进动(precision)
- 第二项：横向弛豫 T2
- 第三项：纵向弛豫 T1
- $M_0$：平衡磁化强度

---

### 4.2 T1 / T2 弛豫信号模型

组织在 RF pulse 后恢复到平衡状态的过程由两种弛豫描述：

- **T1：纵向恢复**
- **T2：横向衰减**

---

 **(1)T1 恢复(Recovery)**

$$
M_z(t) = M_0 \left(1 - e^{-t/T_1}\right)
$$

含义：

- $M_z(t)$：随时间恢复的纵向磁化
- $T_1$：纵向弛豫时间
- 白质、灰质、脑脊液具有不同的 T1 值，用于对比增强

---

 **(2)T2 衰减(Decay)**

$$
M_{xy}(t) = M_0 e^{-t/T_2}
$$

说明：

- T2 反映横向磁化的衰减速度
- 脑脊液 T2 长 → 信号亮
- 白质 T2 短 → 信号暗

---

 **(3)Proton Density(PD)信号表达**

未强烈依赖 T1/T2 的情况下，信号主要由质子密度(PD)决定：

$$
S_{\text{PD}} \propto \rho \left(1 - e^{-TR/T_1}\right) e^{-TE/T_2}
$$

在 PD 强调序列中：

- 选 **长 TR** → 消除 T1 影响
- 选 **短 TE** → 消除 T2 影响

最终信号约似：

$$
S_{\text{PD}} \approx \rho
$$

---

### 4.3 k-space 采样与重建

MRI 数据首先采样在 **k-space(频域)**  中，而不是直接得到图像。k-space 采样与二维傅里叶变换直接相关。

---

 **(1)MRI 信号方程(二维傅里叶编码)**

物体磁化分布 $\rho(x,y)$ 在频率编码与相位编码梯度作用下，采样信号为：

$$
S\left(k_x, k_y\right)=\iint \rho(x, y) e^{-j 2 \pi\left(k_x x+k_y y\right)} d x d y
$$

其中：

- $S(k_x,k_y)$：采集到的 k-space 数据
- $k_x, k_y$：由梯度磁场决定的编码量
- $\rho(x,y)$：被成像组织的质子密度或复合信号

这是 **二维傅里叶变换** 的标准形式。

---

 **(2)逆 Fourier 重建**

图像恢复通过对 k-space 做逆傅里叶变换：

$$
\rho(x, y)=\iint S\left(k_x, k_y\right) e^{j 2 \pi\left(k_x x+k_y y\right)} d k_x d k_y
$$

数值实现：

$$
\rho = \mathcal{F}^{-1}\{ S \}
$$

临床 MRI 设备中采用快速傅里叶变换(FFT)。

## 5. 超声成像（Ultrasound）关键公式

超声成像以机械波的传播、反射与散射为基础，其信号处理链路相比 CT/MRI 更依赖波动方程、回波强度及包络检测。本节总结超声 B-mode 成像中最重要的数学公式。

---

### 5.1 声波传播基本关系

超声本质是纵波(压力波)，在组织中传播时，其速度、频率和波长存在基本物理关系。

---

 **(1)波速、频率与波长**

声速 $c$、频率 $f$、波长 $\lambda$ 的关系为：

$$
c = \lambda f
$$

典型组织声速：

|组织|声速 (m/s)|
| --------------| ----------------|
|脂肪|\~1450|
|肌肉|\~1580|
|软组织平均值|**1540**(临床常用值)|

---

 **(2)声压波表达式(1D)**

传播中的声波可表示为：

$$
p(x,t) = p_0 \cos (2\pi f t - kx)
$$

其中：

- $k = \frac{2\pi}{\lambda}$ 为波数
- $p_0$ 为压力振幅

---

### 5.2 回波形成机制

超声图像亮度主要取决于界面反射与组织衰减。

---

 **(1)组织衰减模型**

声波在传播距离 $d$ 后，振幅衰减为：

$$
A(d) = A_0 e^{-\alpha d}
$$

或以 dB 表示(更常见)：

$$
A_{\mathrm{dB}}(d) = A_{\mathrm{dB}}(0) - \alpha_{\mathrm{dB}} d
$$

其中：

- $\alpha$：线性衰减系数
- $\alpha_{\mathrm{dB}}$：以 dB/cm 表示的衰减常数

衰减随频率上升而增加，因此高频分辨率更好但穿透性更差。

---

 **(2)反射系数(Acoustic Impedance)**

不同组织界面的反射强度由声阻抗决定：

$$
Z = \rho c
$$

声阻抗差异越大，反射越强。

界面反射系数为：

$$
R = \left( 
\frac{Z_2 - Z_1}{Z_2 + Z_1}
\right)^2
$$

其中：

- $Z_1, Z_2$：两种组织的声阻抗
- $R$：反射强度比例(0\~1)

这是 B-mode 成像亮度的最基础来源。

---

### 5.3 B-mode 成像数学

B-mode(Brightness mode)是最常见的超声成像方式，涉及回波包络提取与对数压缩。

---

 **(1)包络检测(Hilbert 变换)**

接收信号通常为射频信号 $x(t)$，需要提取其包络：

$$
A(t) = \sqrt{ x^2(t) + \hat{x}^2(t) }
$$

其中：

- $\hat{x}(t)$：$x(t)$ 的 Hilbert 变换
- $A(t)$：包络(代表强度)

包络代表界面反射的强度，是 B-mode 灰度图的核心。

---

 **(2)B-mode 对数压缩(Log Compression)**

原始 RF 包络的动态范围非常大(\>60 dB)，需要压缩到可显示范围：

$$
I_{\text {display }}=\log (1+A(t))
$$

或常见的 dB 形式：

$$
I_{\mathrm{dB}} = 20 \log_{10}(A)
$$

对数压缩可：

- 压缩动态范围
- 增强弱反射组织的可见性
- 提高整体对比度

---

 **(3)扫描线(Scanline)到图像**

B-mode 最终图像是由多条 A-scan 组成。

数学上可视为：

$$
I(x,y) = \text{LogCompress}\left( A(t) \right)
$$

其中 $x$ 由扫描角度决定，$y$ 为深度(传播时间)。

## 6. PET / SPECT 成像关键公式

PET(Positron Emission Tomography)与 SPECT(Single Photon Emission Computed Tomography)基于放射性核素衰变产生的γ光子来成像，其数学本质包括放射性衰变模型、泊松统计前向模型以及迭代重建(MLEM / OSEM)。本节给出 PET/SPECT 常用的核心公式。

---

### 6.1 放射性衰变

PET/SPECT 的信号源是放射性核素，其示踪剂活度遵循指数衰减。

---

 **(1)放射性衰减公式**

$$
N(t) = N_0 e^{-\lambda t}
$$

其中：

- $N(t)$：时间 $t$ 的核素数量
- $N_0$：初始核素数量
- $\lambda$：衰变常数，与半衰期相关

半衰期公式：

$$
T_{1/2} = \frac{\ln 2}{\lambda}
$$

---

 **(2)活度(Activity)**

衰变率(每秒衰变次数)为：

$$
A(t) = \lambda N(t)
$$

单位为贝可(Bq)。

---

### 6.2 PET 前向模型（Forward Model）

PET/SPECT 探测到的投影数据本质上是**放射性衰变 → γ光子发射 → 探测器响应**的统计采样，通常遵循​**泊松分布**。

---

 **(1)泊松统计模型**

探测器 bin $i$ 的计数服从泊松分布：

$$
y_i \sim \text{Poisson}(\lambda_i)
$$

其中：

- $y_i$：实际观测计数(sinogram 的一个像素)
- $\lambda_i$：期望计数

---

 **(2)投影数据形成方程**

PET 的前向投影可表示为：

$$
\lambda_i = \sum_{j} a_{ij} x_j + r_i
$$

其中：

- $x_j$：像素 $j$ 处的放射性活度(待重建)
- $a_{ij}$：系统矩阵(system matrix)

  - 包含几何因素、衰减、散射、检测效率
- $r_i$：随机事件、散射事件、背景噪声

向量形式：

$$
\boldsymbol{\lambda} = A \mathbf{x} + \mathbf{r}
$$

这是 PET 重建的基础。

---

### 6.3 PET/SPECT 重建（MLEM / OSEM）

PET/SPECT 多采用基于统计的 ​**MLEM 或 OSEM 迭代重建**，源于最大似然估计(MLE)。

---

 **(1)MLE(Maximum Likelihood Estimation)**

观测数据 $y_i$ 的似然函数：

$$
L(\mathbf{x}) = \prod_{i} 
\frac{\lambda_i^{y_i} e^{-\lambda_i}}{y_i!}
$$

其中 $\lambda_i = \sum_j a_{ij} x_j$。

对数似然：

$$
\log L(\mathbf{x}) = \sum_i \left[y_i \ln \lambda_i - \lambda_i\right] + C
$$

MLE 目标：

$$
\hat{\mathbf{x}}
= \arg\max_{\mathbf{x}} \log L(\mathbf{x})
$$

---

 **(2)MLEM(Expectation–Maximization)更新公式**

经典的 MLEM 更新为：

$$
x_j^{(k+1)}=x_j^{(k)}\frac{\sum_i a_{ij} \frac{y_i}{\sum_m a_{im}x_m^{(k)}}}{\sum_i a_{ij}}
$$

其中：

- 第一个分数：数据一致性项
- 分母：归一化因子
- 形式类似于“前投影 → 归一化 → 反投影”

MLEM 收敛性好但速度慢。

---

 **(3)OSEM(Ordered Subsets Expectation Maximization)**

OSEM 将 sinogram 分成多个子集(subsets)，每次只用部分投影更新，提高速度。

OSEM 更新形式：

$$
x_j^{(k+1)}=x_j^{(k)}\frac{\sum_{i \in S_k} a_{ij} \frac{y_i}{\sum_m a_{im}x_m^{(k)}}}{\sum_{i \in S_k} a_{ij}}
$$

其中 $S_k$ 是第 $k$ 个子集。

- 加快重建速度(约快 MLEM 的 S 倍)
- 临床 PET/SPECT 重建常用 OSEM

## 7. X-ray / DR 成像关键公式

X-ray / DR(Digital Radiography)成像的核心来自光子穿透组织时的衰减与探测器响应。其物理模型通常可由 **Lambert-Beer 定律 + 线积分投影模型 + 探测器增益校正** 构成。本节总结 DR 成像中最重要的公式。

---

### 7.1 投影模型（线积分）

在 DR(投影摄影)中，X 射线从单一方向穿过物体，穿透路径上的衰减构成二维影像像素值。

---

 **(1)光子衰减模型(Lambert–Beer)**

对于入射强度 $I_0$，透射强度 $I$ 为：

$$
I = I_0 \exp
\left(
-\int_L \mu(s) \, ds
\right)
$$

其中：

- $\mu(s)$：路径上位置 $s$ 处的衰减系数
- $L$：光线穿过的路径

---

 **(2)投影(线积分)模型**

定义投影：

$$
p = \int_L \mu(s)\, ds
$$

则 DR 图像的理想亮度模型可表示为：

$$
I = I_0 e^{-p}
$$

log-transform 后得到线性表达式：

$$
p = -\ln \left(\frac{I}{I_0}\right)
$$

意义：

- DR 是 Radon 变换的单角度特例(CT 是多角度)
- p 反映 X 射线的整体衰减量，决定成像亮度

---

 **(3)亮度与衰减关系(直接表示)**

亮度(透过率)通常与衰减呈负相关：

$$
\text{Brightness}(x,y) \propto e^{-\mu(x,y) d}
$$

其中 $d$ 为组织厚度。

---

### 7.2 图像归一化与增益校正

真实探测器存在不均匀性、暗电流、增益偏差，因此必须对 DR 图像进行平场(Flat-field)校正，以恢复正确的衰减印记。

---

 **(1)探测器响应模型**

原始 DR 图像可表示为：

$$
I_{\text{raw}}=G \cdot I_{\text{signal}} + D
$$

其中：

- $G$：像素增益(gain)，随像素变化
- $D$：暗电流(dark field)
- $I_{\text{signal}}$：理想信号

---

 **(2)平场校正(Flat-field Correction)**

平场校正的公式为：

$$
I_{\text{corr}}=\frac{ I_{\text{raw}} - I_{\text{dark}} }     { I_{\text{flat}} - I_{\text{dark}} }
$$

其中：

- $I_{\text{raw}}$：原始采集图像
- $I_{\text{flat}}$：均匀照射下的平场图
- $I_{\text{dark}}$：无照射时的暗场图

意义：

- 校正增益不均匀(pixel gain variation)
- 校正暗电流偏置
- 恢复正确的射线透射信息

---

 **(3)线性归一化(用于可视化)**

校正后图像常做归一化：

$$
I_{\text{norm}} = 
\frac{I_{\text{corr}} - \min(I_{\text{corr}})}
     {\max(I_{\text{corr}})-\min(I_{\text{corr}})}
$$

用于：

- 显示优化
- 后续图像处理 / 机器学习算法

## 8. 图像处理与增强的常用公式

医学影像处理中，预处理、增强和特征计算是 CT、MRI、超声、PET、X-ray 等各模态共同依赖的基础步骤。本节总结常用的标准化方法、滤波器、边缘算子，以及重采样中的仿射变换与插值公式。

---

### 8.1 直方图与归一化（Normalization）

标准化是提高图像可比性与模型鲁棒性的关键步骤。

---

 **(1)Min–Max 归一化**

将值线性映射到 $[0,1]$：

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

适用于灰度归一、窗宽映射(如 CT 图像)。

---

 **(2)Z-score 标准化**

常用于深度学习模型的输入标准化：

$$
x' = \frac{x - \mu}{\sigma}
$$

其中：

- $\mu$：均值
- $\sigma$：标准差

使数据满足零均值、单位方差。

---

 **(3) 直方图均衡化(Histogram Equalization)**

通过增强对比度，使灰度分布更均匀。

CDF(累计分布函数)为：

$$
\text{CDF}(x)=\sum_{i=0}^{x} \frac{h(i)}{N}
$$

均衡化后的像素：

$$
x' = (L-1) \cdot \text{CDF}(x)
$$

其中：

- $h(i)$：灰度 i 的频率
- $N$：像素总数
- $L$：灰度级数(通常 256)

适用于 DR、超声、部分 MRI 场景。

---

### 8.2 滤波与特征计算

图像滤波用于平滑、去噪、边缘增强，是图像处理的基础操作。

---

 **(1)高斯滤波(Gaussian Filter)**

一维高斯核：

$$
G(x) = 
\frac{1}{\sqrt{2\pi\sigma^2}}
e^{ -\frac{x^2}{2\sigma^2} }
$$

二维高斯核：

$$
G(x,y) =
\frac{1}{2\pi\sigma^2}
e^{ -\frac{x^2+y^2}{2\sigma^2} }
$$

作用：

- 平滑图像
- 减少高频噪声
- 常用于去噪前处理(如 CT、X-ray、Ultrasound)

---

 **(2)Sobel 边缘算子**

用于检测水平或垂直边缘。

Sobel-x：

$$
G_x =
\left[\begin{array}{ccc}
-1 & 0 & 1\\
-2 & 0 & 2\\
-1 & 0 & 1
\end{array}\right]
$$

Sobel-y：

$$
G_y =
\left[\begin{array}{ccc}
-1 & -2 & -1\\
0 & 0 & 0\\
1 & 2 & 1
\end{array}\right]
$$

梯度强度：

$$
|\nabla I| = \sqrt{ (G_x * I)^2 + (G_y * I)^2 }
$$

适用于边缘检测、形态分析。

---

 **(3)Laplacian 边缘增强**

二阶导数算子：

$$
\nabla^2 I = 
\frac{\partial^2 I}{\partial x^2}
+
\frac{\partial^2 I}{\partial y^2}
$$

典型离散模板：

$$
\left[\begin{array}{ccc}
0 & -1 & 0\\
-1 & 4 & -1\\
0 & -1 & 0
\end{array}\right]
$$

可用于锐化或边缘增强。

---

### 8.3 重采样与对齐（Resampling & Registration）

医学图像对齐常基于仿射变换与空间插值。

---

 **(1)仿射变换矩阵**

三维仿射变换(常用于 CT/MRI 注册)：

$$
\mathbf{x}' = \mathbf{A}\mathbf{x} + \mathbf{b}
$$

齐次形式：

$$
\left[\begin{array}{c}
\mathbf{x}' \\
1
\end{array}\right]=
\left[\begin{array}{cc}
\mathbf{A} & \mathbf{b}\\
0 & 1
\end{array}\right]
\left[\begin{array}{c}
\mathbf{x} \\
1
\end{array}\right]
$$

其中：

- $\mathbf{A}$：包含旋转、缩放、剪切
- $\mathbf{b}$：平移向量

用于：

- 多模态配准(MRI ↔ CT)
- 图像对齐
- 数据标准化(e.g., 统一 voxel spacing)

---

 **(2)三线性插值(Trilinear Interpolation)**

重采样三维体数据最常用的方法。

三线性插值的通式：

$$
f(x,y,z)=
\sum_{i=0}^1\sum_{j=0}^1\sum_{k=0}^1
f(i,j,k)
(1-|x-i|)
(1-|y-j|)
(1-|z-k|)
$$

含义：

- 每个点由周围 8 个体素加权求得
- 权重由距离线性决定

适用于：

- 体数据重采样
- CT/MRI 归一化 spacing
- 空间变换中的 resample 步骤

## 9. 深度学习中的关键公式（医学影像常用）

深度学习已广泛应用于医学影像的分割、分类、检测和重建任务。本节总结在医学影像任务中最常用的卷积公式、损失函数和评价指标，为后续算法理解提供数学基础。

---

### 9.1 卷积层（2D / 3D）

卷积是 CNN 在医学影像中最重要的运算(例如 2D MRI 切片、3D CT 体数据、超声序列等)。

---

 **(1)卷积运算公式**

● 2D 卷积(图像)

$$
y(i,j) = \sum_m \sum_n x(i-m, j-n)\, k(m,n)
$$

● 3D 卷积(体数据)

$$
y(i,j,k) = 
\sum_{u} \sum_{v} \sum_{w}
x(i-u, j-v, k-w)\, k(u,v,w)
$$

3D 卷积广泛用于 CT/MRI 分割和 3D 检测任务。

---

 **(2)卷积输出尺寸计算公式**

● 2D 输出尺寸

$$
H_{\text{out}} = 
\frac{H_{\text{in}} - K + 2P}{S} + 1
$$

$$
W_{\text{out}} = 
\frac{W_{\text{in}} - K + 2P}{S} + 1
$$

● 3D 输出尺寸

$$
D_{\text{out}} =
\frac{D_{\text{in}} - K + 2P}{S} + 1
$$

参数说明：

- $K$：kernel size
- $S$：stride
- $P$：padding
- 输入/输出维度用于 CNN 结构设计(UNet、VNet 等)

---

### 9.2 损失函数（Loss Functions）

医学影像任务常涉及类别不平衡，因此 Dice Loss、Focal Loss 在分割与检测任务中尤其常用。

---

 **(1)Dice Loss(分割常用)**

Dice 系数：

$$
\text{Dice} =
\frac{2|A \cap B|}{|A|+|B|}
$$

Dice Loss：

$$
\mathcal{L}_{Dice} = 1 - \text{Dice}
$$

用于分割任务(尤其是器官、病灶、小目标)。

---

 **(2)Cross-Entropy Loss(分类/分割基础)**

● 二分类交叉熵

$$
\mathcal{L}_{CE}=
- \left[ 
y \log(\hat{y}) + (1-y)\log(1-\hat{y})
\right]
$$

● 多分类交叉熵

$$
\mathcal{L}_{CE}=
-\sum_{c=1}^{C}
y_c \log(\hat{y}_c)
$$

---

 **(3)Focal Loss(解决类别不平衡)**

Focal Loss 抑制简单样本，突出困难样本：

$$
\mathcal{L}_{Focal}=
-(1-\hat{y})^\gamma \, y\log(\hat{y})
$$

其中：

- $\gamma$：调节难样本的权重(典型值 1–3)
- 常用于肺结节检测、肿瘤检测等类别极不平衡任务

---

### 9.3 评价指标（Segmentation & Classification Metrics）

评价指标反映模型在医学影像任务中的性能，尤其是在病灶分割、器官分割和病理分类中尤为关键。

---

 **(1)Dice 系数**

$$
\text{Dice} =
\frac{2|A \cap B|}{|A|+|B|}
$$

- 0(差) → 1(完美)
- 常用于 CT/MRI 器官/病灶分割评估

---

 **(2)IoU(Intersection over Union)**

$$
\text{IoU}=
\frac{|A \cap B|}{|A \cup B|}
$$

与 Dice 关系：

$$
\text{Dice} = \frac{2\text{IoU}}{1+\text{IoU}}
$$

---

 **(3)Sensitivity(敏感度)**

$$
\text{Sensitivity}=
\frac{TP}{TP+FN}
$$

衡量“发现病灶”的能力。

---

 **(4)Specificity(特异度)**

$$
\text{Specificity}=
\frac{TN}{TN+FP}
$$

衡量“避免误报”的能力。

‍
