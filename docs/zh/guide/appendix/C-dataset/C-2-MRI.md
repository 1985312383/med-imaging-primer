---
title: 附录C-2：MRI 类数据集
description: 介绍 MRI 类数据集
---

# 附录C-2：MRI 类数据集

## 1.BraTS（脑肿瘤 MRI 分割数据集）

### 1.1.BraTS 简介

**BraTS（Brain Tumor Segmentation Challenge）**  是 MICCAI 自 2012 起持续推出的脑胶质瘤多模态 MRI 分割数据集，也是医学影像分割最重要的公开基准之一。其数据来自多中心，提供统一预处理和专家精细标注，用于评估肿瘤检测、区域分割、生存期预测等任务，广泛应用于学术研究和临床辅助诊断方法开发。

### 1.2.BraTS 数据结构

典型版本（如 BraTS 2018–2021）包含数百到上千例病例，每例提供四种标准化 MRI 序列（均为 NIfTI 格式并已 skull-stripping 与空间对齐）：

- **T1、T1CE、T2、FLAIR**

标注为三类体素级肿瘤区域：

- **WT（整体肿瘤）**
- **TC（肿瘤核心）**
- **ET（增强肿瘤）**

常见规模示例：BraTS 2020 含 ​**369 例训练 + 125 验证 + 166 测试**​；BraTS 2021 扩展至 ​**1500+ 例**。每例数据结构稳定，适用于分割、分类和预后建模。

### 1.3.BraTS 下载方式

- 官方网站（各年份入口）：https://www.med.upenn.edu/cbica/brats/
- Kaggle 镜像：[Search | Kaggle](https://www.kaggle.com/search?q=BRaTS+in%3Adatasets)
- Aistudio下载（BraTS2015）：https://aistudio.baidu.com/datasetdetail/26367

## 2.OASIS（Open Access Series of Imaging Studies）

### 2.1 OASIS 简介

OASIS 是由 Washington University in St. Louis (WashU) 发起的一个脑影像数据集系列，旨在向科研社区免费提供包含正常老化及认知衰退（如 Alzheimer’s disease）人群的 MRI、PET、临床与认知数据。  
该系列包括多个子集（OASIS-1、OASIS-2、OASIS-3、OASIS-4），覆盖横断面与纵向、多模态影像数据，适合研究脑结构变化、老化、认知衰退、影像-临床关联等。

### 2.2 OASIS 数据结构

以下为各子集的主要结构与特点：

#### ​**OASIS-1**（Cross-sectional）

- 含 **416 名受试者**（年龄 18-96 岁）及 100 名 60 岁以上轻度至中度 Alzheimer 病例。
- 每名受试者在一次扫描中取得 3 或 4 帧 T1 加权 MRI。
- 数据格式公开，提供影像、人口统计、认知评分等。

#### ​**OASIS-2**（Longitudinal）

- 含 **150 名参与者**（年龄 60-96 岁），总共 **373 次扫描会话**。
- 纵向设计：每位参与者在两次或多次访问中扫描，间隔至少一年。
- 旨在研究随时间变化的脑结构和认知状态。

#### ​**OASIS-3**（Longitudinal Multimodal Neuroimaging）

- 包含约 **1,378 名参与者**（其中 755 名认知正常，622 名处于不同认知衰退阶段，年龄约 42-95 岁）
- 包含 **2,842 次 MR 会话**（包含 T1w、T2w、FLAIR、ASL、SWI、resting-state BOLD、DTI）
- 包含 **2,157+ 次 PET 扫描**（如 AV45、FDG）、以及额外子项目如 “OASIS-3\_AV1451”（Tau PET）
- 提供丰富的多模态影像 + 临床 +认知 +生物标志物数据，是老化与 Alzheimer 研究的重要资源。

#### ​**OASIS-4**（Clinical Cohort）

- 包含 **663 名受试者**（21-94 岁），主要为有记忆障碍或痴呆评估的临床人群。
- 与 OASIS-3 不同，是一个独立的临床队列，而非 OASIS-3 的延续。

#### **共同特点 & 数据格式**

- 所有影像数据均已去识别化（de-identified）。
- 提供MRI原始数据（通常为 DICOM/NIfTI 格式）及处理后结构（如 FreeSurfer 分割）等。
- 配有临床／认知／人口统计元数据，如年龄、性别、MMSE 或 CDR 等评分。

### 2.3 OASIS 下载方式

- 官方主页： https://sites.wustl.edu/oasisbrains/（列出 OASIS-1、2、3、4 数据项）
- 注册／申请访问：部分子集需在 XNAT 或 NITRC‑IR 平台进行账号注册与数据使用协议同意。
- kaggle：[Search | Kaggle](https://www.kaggle.com/search?q=OASIS+in%3Adatasets)

## 4.fastMRI（加速 MRI 重建数据集）

### 4.1.fastMRI 简介

fastMRI 是由 NYU Langone Health 与 Meta AI Research（前 Facebook AI Research）合作推出的公开医学影像数据集，目标是通过 AI 方法探索 **加速 MRI 扫描、减少采样时间、提升重建质量**。  
数据集具有原始 k-空间（raw k-space）数据＋DICOM 重建图像，涉及膝关节、脑部、前列腺、乳腺等多个器官，因其 “真实原始 MRI 测量 + 多模态” 特性，在医学影像重建、压缩采样、迁移学习、跨器官泛化研究中影响深远。

### 4.2.fastMRI 数据结构

（1）涵盖器官与模态

- 膝关节（Knee）MRI：超过 \~1,500 例完全采样 + 10,000 例临床 DICOM 图像。
- 脑部（Brain）MRI：约 6,970 例完全采样（1.5 T / 3 T）包括 T1、T2、FLAIR 等序列。
- 前列腺（Prostate）MRI：312 例 3 T 获取的 axial T2 + DWI 序列。
- 乳腺（Breast）MRI：300 例 3 T 动态对比增强（DCE）MR，使用 radial k-space 采样。

（2）数据格式与标注

- 提供 **原始 k-space 数据**（ISMRMRD 或 vendor-neutral 格式） + **重建 DICOM/NIfTI 图像**。
- 已进行去识别化处理（metadata 我保护/清理），每个子集依据协议申请使用。
- 标注方面：虽然主 focus 是重建任务，但也衍生出 fastMRI+ 子集，包含膝/脑部病灶的专家 bounding box 注释。

（3）任务类型

- 加速 MRI 重建：在少量 k-space 采样下恢复高质量图像。
- 压缩采样与重建算法基准：提供标准评价指标（如 PSNR、SSIM）以比较不同方法。
- 跨器官迁移学习、模型泛化与弱监督学习（借助 fastMRI+ 注释）

### 4.3.fastMRI 下载方式

- 官方主页： https://fastmri.med.nyu.edu/
- AWS 开放数据存储：https://registry.opendata.aws/nyu-fastmri/
- GitHub 代码仓库： https://github.com/facebookresearch/fastMRI（含数据加载器、baseline 模型）

​**申请流程**：

- 需同意 “Data Sharing Agreement / Dataset Sharing Agreement”
- 填写机构信息、研究用途等
- 数据仅限 “研究或教学用途” 且禁止未经授权的再分发。

‍
