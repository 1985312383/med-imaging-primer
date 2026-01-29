---
title: 附录C-1：CT 类数据集
description: 介绍 CT 类数据集
---

# 附录C-1：CT 类数据集

## 1.LIDC-IDRI（肺结节CT数据库）

### 1.1.LIDC-IDRI 简介

介绍论文： [The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041807/)

The Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI) 是一个经典的肺部结节CT影像数据库，用于肺结节的检测、良恶性判断等计算机辅助诊断研究。该数据库由多个北美研究机构和影像设备厂商合作创建，包含经过多名放射科医师标注的低剂量胸部CT扫描，是肺结节检测算法评估的标准数据集之一。

### 1.2.LIDC-IDRI 数据结构

**数据内容与结构:**  数据集收集了**1018例**胸部CT扫描，每例都有对应的标注文件。每个CT由多位放射科医师独立标记肺内的结节，包括结节所在位置的轮廓（边界）以及结节直径、形状特征和恶性可能等级等（标注以XML格式提供）。所有≥3 mm的结节都被标注，并附有医师对结节恶性概率的评分；对于一些\<3 mm的小结节和疑似伪影，XML中也有记录但未作全面标注。通过处理这些标注，可得到每例CT中结节的像素级分割掩码及属性。

### 1.3.LIDC-IDRI 下载方式

数据通过癌症影像档案（[TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/)）以合集形式提供，任何注册用户均可免费获取。登录TCIA网站后，在LIDC-IDRI项目页面即可按患者或影像批量下载DICOM格式的CT影像及对应的XML标注。

TCIA还提供了NBIA Data Retriever工具方便批量下载。除了官方渠道，社区也提供了镜像下载源：例如Kaggle上有用户整理的LIDC-IDRI子集，国内科研者可通过百度Aistudio的共享项目获取⭐**Aistudio下载** **[Part1](https://aistudio.baidu.com/aistudio/datasetdetail/63957)**  **[Part2](https://aistudio.baidu.com/aistudio/datasetdetail/64008)**

## 2.LiTS（肝脏肿瘤CT分割数据集）

### 2.1.LiTS 简介

Liver Tumor Segmentation Benchmark (LiTS) 是用于肝脏及肝脏肿瘤自动分割任务的公开CT数据集。该数据最初于2017年在医学影像计算大会(MICCAI)的LiTS挑战赛上发布，旨在比较不同算法对CT影像中肝脏和肝内肿瘤的分割性能。目前LiTS已成为肝脏肿瘤分割领域的常用基准数据。

### 2.2.LiTS 数据结构

数据集包含**131组**腹部CT扫描的训练集，以及**70组**CT扫描的测试集。训练集中每例CT都由专业放射科医生手工勾画了**肝脏轮廓和肝脏内肿瘤**的体素级分割标注，标注以nii体积文件或mhd+raw格式提供。测试集70例CT的影像公开可下载，但其对应的肝/肿瘤标注未公开，用于线上评测模型性能（提交结果至官网评测系统）。需要注意的是，LiTS训练数据实际上已包含了早期3D-IRCADb等肝脏数据集的所有病例；因此研究中不应简单将LiTS与这些数据集混合，以免重复。同样，**Medical Segmentation Decathlon**挑战中的肝脏分割任务直接使用的就是LiTS的数据。总体而言，LiTS提供了丰富的肝脏肿瘤CT样本，病例来自多中心，涵盖不同肿瘤大小和CT成像质量。

### 2.3.LiTS  下载方式

- github：[Auggen21/LITS-Challenge: Liver Tumor Segmentation Challenge](https://github.com/Auggen21/LITS-Challenge)
- kaggle：[LITS Dataset](https://www.kaggle.com/datasets/harshwardhanbhangale/lits-dataset)
- Aistudio下载：[LiTS肝脏/肝肿瘤分割_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/dataset/detail/10273/intro)

## 3.KiTS19（肾脏肿瘤CT分割数据集）

### 3.1.KiTS19 简介

Kidney Tumor Segmentation 2019 (KiTS19) 数据集是在2019年MICCAI肾脏肿瘤自动分割挑战中发布的腹部CT影像集。该数据集旨在推动肾脏及肾肿瘤的语义分割算法研发，帮助评估不同方法在检测和分割肾脏肿瘤方面的效果。KiTS19因标注精细、病例数量较多，已成为医学影像分割领域的重要公开数据资源。

### 3.2.KiTS19 数据结构

KiTS19收录了**210例**肾肿瘤患者术前的对比增强腹部CT扫描，以及对应的人工分割标注。每例CT均提供了对**左右肾脏轮廓**和**肾脏内肿瘤**的逐像素标签，由临床专家根据术后病理确认结果手工勾画。CT影像以DICOM格式发布，分辨率约在0.5–0.8mm，标注以与原CT同维的掩膜（可下载为包含多个Label的DICOM SEG文件或nii文件）。KiTS19最初划分为训练集210例（带标签）和**测试集90例**（仅影像无标签，用于竞赛评估），总计约300例CT数据。训练集中还提供每例患者的部分临床信息（如手术类型、术后预后等），以支持综合分析。

#### 3.3.KiTS19 下载方式

- [Data - KiTS19 - Grand Challenge](https://kits19.grand-challenge.org/data/)
- github：[neheller/kits19: The official repository of the 2019 Kidney and Kidney Tumor Segmentation Challenge](https://github.com/neheller/kits19)
- kaggle：[KITS 19 - Kidney Tumor Segmentation](https://www.kaggle.com/datasets/orvile/kits19-png-zipped)
- aistudio：[Kits19肾脏肿瘤分割_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/24582)

## 4.DeepLesion（大规模 CT 病灶检测数据集）

### 4.1.DeepLesion 简介

**DeepLesion** 是 NIH 在 2018 年发布的超大规模 **泛器官 CT 病灶检测数据集**，来源于真实临床 PACS 系统中医生的测量记录（RECIST 直径标注）。不同于仅针对某一器官的传统数据集，DeepLesion 包含从胸腔到腹盆腔等多部位的病灶，类型多样、噪声真实，极具临床复杂性。凭借 **32k+ 病灶、10k+ CT 检查、4k+ 患者** 的规模，它已成为病灶检测、弱监督学习、跨器官泛化、异常筛查等任务的核心 benchmark。

### 4.2.DeepLesion 数据结构

数据集共包含 ​**32,120 个病灶标注**​，覆盖 ​**10,594 次 CT 检查**（来自 4,427 名患者）。每个病灶提供：

- **2D bounding box（x1,y1,x2,y2）**
- 所在切片号、病灶直径（RECIST）
- 对应预处理后的 **512×512 PNG CT 切片**

标注来自临床医生实际测量点，因此真实但可能存在轻微误差，适合研究弱监督与鲁棒学习。  
此外，通过自然语言处理自动生成 ​**8 类弱解剖标签**​：骨、肝、肺、淋巴结、软组织、肾脏、盆腔、其他（非严格人工标注）。  
官方推荐按 ​**患者级划分**：训练集 70%，验证集 15%，测试集 15%，避免数据泄漏。

DeepLesion 的病例涵盖增强与非增强扫描，层厚、成像参数多样，真实度高，有助于开发对不同扫描条件具有泛化能力的模型。

### 4.3.DeepLesion 下载方式

- 官方下载：https://datasetninja.com/deep-lesion
- 官方论文（推荐引用）：Yan et al., *DeepLesion: Automated Deep Mining, Categorization and Detection of Significant Radiology Image Findings*, CVPR 2018.  
  https://arxiv.org/abs/1710.01766
- Kaggle：https://www.kaggle.com/datasets/kmader/nih-deeplesion-subset
