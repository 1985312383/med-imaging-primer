---
title: 附录C-3：X-ray 类数据集
description: 介绍 X-ray 类数据集
---

# 附录C-3：X-ray 类数据集

## 3.CheXpert（胸部 X 光大规模病灶检测数据集）

### 3.1.CheXpert 简介

CheXpert 是由 Stanford University 医学影像与 AI 团队于 2019 年发布的公开胸部 X 光影像数据集，主要用于胸腔疾病（如肺炎、胸腔积液、心脏扩大等）的多标签预测。数据集收录自临床真实场景，强调不确定（uncertainty）标签的处理，被广泛作为胸片 AI 研究的基准。

### 3.2.CheXpert 数据结构

- 收录 **224,316 张胸部 X 光片**，来自 **65,240 名患者**，时间范围约为 2002 年 10 月至 2017 年 7 月。
- 每张影像配有对应的放射科报告，并从报告中自动提取 **14 个观察指标（observations）** ，标签分为「阳性（1）」「阴性（0）」「不确定（–1/ u）」三类。
- 14 个观察指标包括：atelectasis（肺不张）、cardiomegaly（心脏扩大）、consolidation（实变）、edema（水肿）、enlarged cardiomediastinum、fracture、lung lesion、lung opacity、pleural effusion、pleural other、pneumonia、pneumothorax、support devices、no finding。
- 数据视角：当有多个视图（如正位＋侧位）时，模型通常取各视图预测的最大值作为指标。
- 数据划分：官方提供训练集 + 验证集；测试集由 500 个独立研究（studies）组成，由五位认证放射科医师标注作为参考标准。

### 3.3.CheXpert 下载方式

- 官方主页：https://stanfordmlgroup.github.io/competitions/chexpert/
- 访问条件：通常需注册账户、同意研究使用协议（Research Use Agreement, RUA）后，才能下载数据。
- Kaggle 镜像（例如 “CheXpert-v1.0-small”）也可访问作为子集。https://www.kaggle.com/datasets/ashery/chexpert

## 2.MIMIC-CXR（大规模胸片公开数据集）

### 2.1 MIMIC-CXR 简介

MIMIC-CXR 是由 Beth Israel Deaconess Medical Center（BIDMC）在波士顿所采集，并由 MIT Laboratory for Computational Physiology 和其它单位整理公开的、**去识别化胸部 X 光（chest radiograph）数据集**。其包含数十万张真实临床胸片影像并匹配放射科报告，面向图像理解、自然语言处理与决策支持研究。  
例如，其首版版本描述为：覆盖 2011-2016 年期间约 65,379 名患者、227,835 次影像检查、377,110 张图像。  
该数据集因其规模大、结构清晰、报告附带，被视为胸片 AI 研究的重要基准。

### 2.2 MIMIC-CXR 数据结构

- **影像数量**：约 377,110 张胸片，关联约 227,835 次影像检查。
- **患者数量**：约 65,379 人。
- **视图类型**：多数检查包含正前位 (frontal) + 侧位 (lateral) 视图。
- **影像格式**：提供 DICOM 格式原始影像（去识别化）及对应报告文本。
- **报告文本**：每次检查配有放射科医生所写的自由文本报告，描述影像所见
- ​**数据标注／派生**：用户可基于报告文本提取结构化标签（如病灶有无、设备位置等）用于分类任务。
- ​**任务类型**：包括胸片异常检测／分类、放射报告-影像配对、影像-文本联合建模。
- **使用协议**：数据已做去识别处理，符合 HIPAA Safe Harbour 要求。

### 2.3 MIMIC-CXR 下载方式

- 官方托管平台：https://physionet.org/content/mimic-cxr/2.1.0/
- 下载流程通常包括：注册账户 → 签署数据使用协议 (Data Use Agreement, DUA) → 审核通过 → 下载。

## 3.NIH ChestX-ray14（胸部 X 光多标签公开数据集）

### 3.1 简介

NIH ChestX-ray14 是由美国 National Institutes of Health（NIH）临床中心发布的公开胸部 X 光影像数据集，最初于 2017 年以 “ChestX-ray8” 名称发布，随后扩展为包含 14 类胸腔常见病变（ChestX-ray14）。该数据集含有十万级别以上的临床胸片，配有自动文本挖掘出的多标签，广泛用于胸片分类、检测、弱监督学习研究。

### 3.2 数据结构

- 图像数：约 **112,120 张正位胸片**，来自 **30,805 名唯一患者**。
- 标签：每张影像配有最多 14 个胸腔病变标签 + “No Finding”类别；标签通过 NLP 从放射报告中提取。
- 多标签任务：每张图可能同时含多种病变（例如水肿 + 肺部浸润 + 心脏扩大） → 属于多标签分类情形。
- 图像格式：PNG 格式（部分 DICOM 版本在 Google Cloud 中可访问）
- 数据划分：训练集 86,524 张、测试集 25,596 张。
- 常见任务：胸腔疾病分类、弱监督定位（少量标注 bounding box）、多标签指标（ROC-AUC）评估。

### 3.3 下载方式

- 官方下载页面：由 NIH Clinical Center 提供的 Box 链接 https://nihcc.app.box.com/v/ChestXray-NIHCC
- Google Cloud 公共存储桶：https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest?hl\=zh-cn
- Kaggle 镜像版本：如 “NIH-Chest-X-rays” 在 Kaggle 上提供。https://www.kaggle.com/datasets/nih-chest-xrays/data
- 使用须知：无需付费、无明显使用限制，但要求注明数据来源、引用原论文。
