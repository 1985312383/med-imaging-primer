---
title: Appendix C-1: CT Type Datasets
description: Introduces CT type datasets
---

# Appendix C-1: CT Type Datasets

# 1. LIDC-IDRI (Lung CT Nodule Database)

## 1.1. LIDC-IDRI Introduction

Introduction paper: [The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3041807/)

The Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI) is a classic lung nodule CT image database, used for computer-aided diagnosis research such as lung nodule detection and benign/malignant determination. This database was created through collaboration between multiple North American research institutions and imaging equipment manufacturers, containing low-dose chest CT scans annotated by multiple radiologists, making it one of the standard datasets for evaluating lung nodule detection algorithms.

## 1.2. LIDC-IDRI Data Structure

**Data Content and Structure:** The dataset collects **1018 cases** of chest CT scans, each with corresponding annotation files. Each CT is independently marked by multiple radiologists for lung nodules, including nodule location contours (boundaries) as well as nodule diameter, shape characteristics, and malignancy possibility levels (annotations provided in XML format). All nodules ≥3 mm are annotated, accompanied by physicians' ratings of nodule malignancy probability; for some small nodules <3 mm and suspected artifacts, XML also contains records but without comprehensive annotation. By processing these annotations, pixel-level segmentation masks and attributes of nodules in each CT case can be obtained.

## 1.3. LIDC-IDRI Download Methods

Data is provided as a collection through the Cancer Imaging Archive ([TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/)), and any registered user can obtain it for free. After logging into the TCIA website, on the LIDC-IDRI project page, DICOM format CT images and corresponding XML annotations can be downloaded by patient or image in batches.

TCIA also provides the NBIA Data Retriever tool for convenient batch downloads. In addition to official channels, the community also provides mirror download sources: for example, Kaggle has user-organized LIDC-IDRI subsets, and domestic researchers can obtain them through Baidu Aistudio shared projects ⭐**Aistudio Download** **[Part1](https://aistudio.baidu.com/aistudio/datasetdetail/63957)** **[Part2](https://aistudio.baidu.com/aistudio/datasetdetail/64008)**

# 2. LiTS (Liver Tumor CT Segmentation Dataset)

## 2.1. LiTS Introduction

Liver Tumor Segmentation Benchmark (LiTS) is a public CT dataset for automatic segmentation tasks of liver and liver tumors. This dataset was initially released at the 2017 Medical Image Computing Conference (MICCAI) LiTS Challenge, aiming to compare the performance of different algorithms in segmenting liver and intrahepatic tumors in CT images. Currently, LiTS has become a common benchmark dataset in the field of liver tumor segmentation.

## 2.2. LiTS Data Structure

The dataset contains **131 sets** of abdominal CT scans for training and **70 sets** of CT scans for testing. In the training set, each CT has been manually delineated by professional radiologists with voxel-level segmentation annotations for **liver contours and intrahepatic tumors**, with annotations provided in nii volume files or mhd+raw format. The test set of 70 CT images is publicly available for download, but their corresponding liver/tumor annotations are not publicly disclosed, used for online evaluation of model performance (submit results to the official evaluation system). It should be noted that LiTS training data actually includes all cases from early 3D-IRCADb and other liver datasets; therefore, in research, LiTS should not be simply mixed with these datasets to avoid duplication. Similarly, the liver segmentation task in the **Medical Segmentation Decathlon** challenge directly uses LiTS data. Overall, LiTS provides rich liver tumor CT samples, with cases from multiple centers, covering different tumor sizes and CT imaging qualities.

## 2.3. LiTS Download Methods

- github: [Auggen21/LITS-Challenge: Liver Tumor Segmentation Challenge](https://github.com/Auggen21/LITS-Challenge)
- kaggle: [LITS Dataset](https://www.kaggle.com/datasets/harshwardhanbhangale/lits-dataset)
- Aistudio download: [LiTS肝脏/肝肿瘤分割_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/dataset/detail/10273/intro)

# 3. KiTS19 (Kidney Tumor CT Segmentation Dataset)

## 3.1. KiTS19 Introduction

Kidney Tumor Segmentation 2019 (KiTS19) dataset is an abdominal CT image collection released at the 2019 MICCAI Kidney Tumor Automatic Segmentation Challenge. This dataset aims to promote the development of semantic segmentation algorithms for kidneys and kidney tumors, helping to evaluate the effectiveness of different methods in detecting and segmenting kidney tumors. KiTS19 has become an important public data resource in the field of medical image segmentation due to its fine annotations and relatively large number of cases.

## 3.2. KiTS19 Data Structure

KiTS19 includes **210 cases** of preoperative contrast-enhanced abdominal CT scans from kidney tumor patients, along with corresponding manual segmentation annotations. Each CT provides pixel-level labels for **left and right kidney contours** and **kidney tumors**, manually delineated by clinical experts based on postoperative pathological confirmation results. CT images are published in DICOM format, with resolution approximately 0.5–0.8mm, with annotations provided as masks of the same dimension as the original CT (can be downloaded as DICOM SEG files containing multiple labels or nii files). KiTS19 was initially divided into a training set of 210 cases (with labels) and **test set of 90 cases** (images only without labels, used for competition evaluation), totaling approximately 300 CT data cases. The training set also provides partial clinical information for each patient (such as surgery type, postoperative prognosis, etc.) to support comprehensive analysis.

### 3.3. KiTS19 Download Methods

- [Data - KiTS19 - Grand Challenge](https://kits19.grand-challenge.org/data/)
- github: [neheller/kits19: The official repository of the 2019 Kidney and Kidney Tumor Segmentation Challenge](https://github.com/neheller/kits19)
- kaggle: [KITS 19 - Kidney Tumor Segmentation](https://www.kaggle.com/datasets/orvile/kits19-png-zipped)
- aistudio: [Kits19肾脏肿瘤分割_数据集-飞桨AI Studio星河社区](https://aistudio.baidu.com/datasetdetail/24582)

# 4. DeepLesion (Large-scale CT Lesion Detection Dataset)

## 4.1. DeepLesion Introduction

**DeepLesion** is a large-scale **pan-organ CT lesion detection dataset** released by NIH in 2018, derived from real clinical PACS system physician measurement records (RECIST diameter annotations). Unlike traditional datasets that only target a specific organ, DeepLesion contains lesions from multiple parts of the body from chest to abdomen and pelvis, with diverse types and realistic noise, possessing high clinical complexity. With a scale of **32k+ lesions, 10k+ CT examinations, 4k+ patients**, it has become a core benchmark for lesion detection, weakly supervised learning, cross-organ generalization, and abnormal screening tasks.

## 4.2. DeepLesion Data Structure

The dataset contains a total of **32,120 lesion annotations**, covering **10,594 CT examinations** (from 4,427 patients). Each lesion provides:

- **2D bounding box (x1,y1,x2,y2)**
- Slice number where located, lesion diameter (RECIST)
- Corresponding preprocessed **512×512 PNG CT slice**

Annotations come from actual clinical physician measurement points, therefore realistic but may have slight errors, suitable for studying weakly supervised and robust learning.  
Additionally, **8 classes of weak anatomical labels** are automatically generated through natural language processing: bone, liver, lung, lymph nodes, soft tissue, kidney, pelvis, other (not strictly manually annotated).  
Official recommendation is **patient-level division**: training set 70%, validation set 15%, test set 15%, to avoid data leakage.

DeepLesion cases cover both enhanced and non-enhanced scans, with diverse slice thickness and imaging parameters, high realism, helping to develop models with generalization ability to different scanning conditions.

## 4.3. DeepLesion Download Methods

- Official download: https://datasetninja.com/deep-lesion
- Official paper (recommended for citation): Yan et al., *DeepLesion: Automated Deep Mining, Categorization and Detection of Significant Radiology Image Findings*, CVPR 2018.  
  https://arxiv.org/abs/1710.01766
- Kaggle: https://www.kaggle.com/datasets/kmader/nih-deeplesion-subset