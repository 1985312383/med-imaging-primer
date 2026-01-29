---
title: 附录C-4：多模态数据集
description: 介绍多模态数据集
---

# 附录C-4：多模态数据集

## 1.ADNI（阿尔茨海默病神经影像学）

### 1.1.ADNI 简介

ADNI 是由 National Institute on Aging（NIA）联合加拿大与美国多个中心于 2004 年启动的、长期、 多中心、 观测性研究，旨在通过结构影像（MRI）、功能影像（PET）、生物标志物（CSF、血液）、认知评估及基因组数据来 **鉴定阿尔茨海默病（AD）及其早期阶段（MCI） 的生物标志物**。  
该数据集因其庞大规模、长期随访、多模态数据共享政策而成为 AD 研究、影像-机器学习、预后建模的重要基准资源。 

### 1.2.ADNI 数据结构

（1）研究分期

ADNI 分为多个阶段（ADNI-1、ADNI-GO、ADNI-2、ADNI-3、ADNI-4）以扩展人群及影像模态：

- ADNI-1（2004–2010）：约 200 名健康对照 + 400 名 MCI + 200 名 AD 患者。
- ADNI-GO / ADNI-2 / ADNI-3：逐步扩展早期 MCI、晚期 MCI、Tau PET、功能影像等。
- ADNI-4（2022 起）：旨在提高研究泛化能力、包括远程采样、更多多中心数据。

（2）数据类型

ADNI 提供以下主要数据类型：

- **影像数据**：MRI（结构像、功能像、扩散像）、PET（如 Aβ、FDG、Tau）
- **生物标志物／生物样本**：血液、CSF、基因组／omics 数据、病理数据
- **认知／临床数据**：包括人口统计、教育年限、MMSE、CDR、神经心理测试、随访状态
- ​**多模态融合数据**：如影像＋认知＋基因＋生物标志物，用于预测疾病进展。

（3）数据规模与访问

- 数据库包含数千名受试者、数万次影像／生物样本测量。
- 数据通过 LONI Image & Data Archive (IDA) 平台提供，需提交申请、签署数据使用协议（DUA）才能访问。
- 研究者可按协议下载 MRI、PET、基因、CSF 等数据，用于科研与教学（但须遵守授权条款）。

### 1.3.ADNI 下载方式

- 官方网站： https://adni.loni.usc.edu/ — 列出研究介绍、数据／样本访问入口。
- 数据访问流程：申请账号 → 签署数据使用协议 → 审核通过后登陆 IDA → 搜索项目 → 下载所需数据。
- 用户指南／数据字典： “ADNI Data User Guide” 提供详细的变量说明、数据格式、访问要求。
- 介绍论文：[Alzheimer&apos;s Disease Neuroimaging Initiative (ADNI)](https://n.neurology.org/content/74/3/201.short)

## 2.TCIA（The Cancer Imaging Archive）

### 2.1.TCIA 简介

TCIA 是由美国 National Cancer Institute (NCI) 支持、由 University of Arkansas for Medical Sciences 承办运营的公开影像数据平台，专注于肿瘤影像（CT、MRI、PET、数字病理等）数据的归档、去识别与共享。  
其目标是：促进影像组学、放射组学、肿瘤-基因组联合研究、AI 算法开发与验证。数据以“collections”（病种/模态/研究专题）形式组织，广泛用于肿瘤检测、分割、治疗反应评估、影像-组学分析。

### 2.2.TCIA 数据结构

（1）组织结构

- 数据按 **Collections（合集）**  进行组织，每个合集通常围绕一个**肿瘤类型**（如肺癌、乳腺癌、脑瘤）、**影像模态**（CT、MRI、PET、数字病理）、或**研究专题**。
- 每个 collection 通常包括若干患者 (subjects)、每患者若干检查 (studies)、每检查若干序列/切片 (series)。

（2）影像模态与附加数据

- 主要模态包括：CT、MRI、PET，也有数字病理 (Whole Slide Images) 及结构化支持数据（例如临床数据、基因组数据、放疗结构、剂量计划等）
- 所有影像基本格式为 DICOM。
- 支持数据：部分合集提供临床随访、治疗信息、影像标注（ROI/segmentation）、基因组数据链接等。

（3）规模指标

- 截止发表时间，TCIA 收录 **超过 30 .9 百万张影像**，约 **37,568 名受试者**。
- 收藏数量不断更新，每个 collection 覆盖的受试者数、模态数、标注丰富度差别较大。

（4）任务适用

- 肿瘤检测／分割：利用 CT/MRI 肿瘤肿块 ROI 数据或病灶标注。
- 影像-组学／放射组学：影像特征 + 临床/基因数据联合建模。
- 多模态融合：影像 +病理 +基因 +治疗响应。
- 算法验证基准：因数据来源多中心、真实临床，适合评估泛化能力。

### 3.3.TCIA 下载方式

- 官方主页： https://www.cancerimagingarchive.net/

‍
