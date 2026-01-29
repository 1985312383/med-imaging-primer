---
title: Appendix C-2: MRI Type Datasets
description: Introduces MRI type datasets
---

# Appendix C-2: MRI Type Datasets

# 1. BraTS (Brain Tumor MRI Segmentation Dataset)

## 1.1. BraTS Introduction

**BraTS (Brain Tumor Segmentation Challenge)** is a multi-modal MRI segmentation dataset for brain gliomas continuously launched by MICCAI since 2012, and also one of the most important public benchmarks in medical image segmentation. Its data comes from multiple centers, providing unified preprocessing and fine expert annotations, used to evaluate tasks such as tumor detection, region segmentation, and survival prediction, widely applied in academic research and clinical auxiliary diagnosis method development.

## 1.2. BraTS Data Structure

Typical versions (such as BraTS 2018–2021) contain hundreds to thousands of cases, each providing four standardized MRI sequences (all in NIfTI format and already skull-stripped and spatially aligned):

- **T1, T1CE, T2, FLAIR**

Annotations are three types of voxel-level tumor regions:

- **WT (Whole Tumor)**
- **TC (Tumor Core)**
- **ET (Enhancing Tumor)**

Common scale examples: BraTS 2020 contains **369 training + 125 validation + 166 test** cases; BraTS 2021 expands to **1500+ cases**. Each case's data structure is stable, suitable for segmentation, classification, and prognostic modeling.

## 1.3. BraTS Download Methods

- Official website (entries for each year): https://www.med.upenn.edu/cbica/brats/
- Kaggle mirror: [Search | Kaggle](https://www.kaggle.com/search?q=BRaTS+in%3Adatasets)
- Aistudio download (BraTS2015): https://aistudio.baidu.com/datasetdetail/26367

# 2. OASIS (Open Access Series of Imaging Studies)

## 2.1 OASIS Introduction

OASIS is a brain imaging dataset series initiated by Washington University in St. Louis (WashU), aiming to provide free MRI, PET, clinical and cognitive data of populations including normal aging and cognitive decline (such as Alzheimer's disease) to the research community.  
This series includes multiple subsets (OASIS-1, OASIS-2, OASIS-3, OASIS-4), covering cross-sectional and longitudinal, multi-modal imaging data, suitable for studying brain structure changes, aging, cognitive decline, imaging-clinical correlations, etc.

## 2.2 OASIS Data Structure

The following are the main structures and characteristics of each subset:

### **OASIS-1** (Cross-sectional)

- Contains **416 subjects** (ages 18-96 years) and 100 cases of mild to moderate Alzheimer's disease over 60 years old.
- Each subject obtains 3 or 4 frames of T1-weighted MRI in one scan.
- Data format is public, providing images, demographics, and cognitive scores.

### **OASIS-2** (Longitudinal)

- Contains **150 participants** (ages 60-96 years), with a total of **373 scanning sessions**.
- Longitudinal design: each participant is scanned during two or more visits, with at least one-year interval.
- Aims to study brain structure and cognitive status changes over time.

### **OASIS-3** (Longitudinal Multimodal Neuroimaging)

- Contains approximately **1,378 participants** (of which 755 are cognitively normal, 622 are at different stages of cognitive decline, ages approximately 42-95 years)
- Contains **2,842 MR sessions** (including T1w, T2w, FLAIR, ASL, SWI, resting-state BOLD, DTI)
- Contains **2,157+ PET scans** (such as AV45, FDG), as well as additional sub-projects such as "OASIS-3_AV1451" (Tau PET)
- Provides rich multi-modal imaging + clinical + cognitive + biomarker data, important resource for aging and Alzheimer's research.

### **OASIS-4** (Clinical Cohort)

- Contains **663 subjects** (ages 21-94), mainly clinical population with memory impairment or dementia evaluation.
- Unlike OASIS-3, it is an independent clinical cohort, not a continuation of OASIS-3.

### **Common Features & Data Format**

- All imaging data are de-identified.
- Provide MRI raw data (usually in DICOM/NIfTI format) and processed structures (such as FreeSurfer segmentation).
- Equipped with clinical/cognitive/demographic metadata, such as age, gender, MMSE or CDR scores.

## 2.3 OASIS Download Methods

- Official homepage: https://sites.wustl.edu/oasisbrains/ (lists OASIS-1, 2, 3, 4 data items)
- Registration/application access: some subsets require account registration and data use agreement consent on XNAT or NITRC‑IR platforms.
- kaggle: [Search | Kaggle](https://www.kaggle.com/search?q=OASIS+in%3Adatasets)

# 4. fastMRI (Accelerated MRI Reconstruction Dataset)

## 4.1. fastMRI Introduction

fastMRI is a public medical imaging dataset jointly launched by NYU Langone Health and Meta AI Research (formerly Facebook AI Research), aiming to explore **accelerating MRI scans, reducing sampling time, and improving reconstruction quality** through AI methods.  
The dataset has raw k-space data + DICOM reconstructed images, involving multiple organs such as knee, brain, prostate, and breast. Due to its "true original MRI measurements + multi-modal" characteristics, it has profound influence in medical image reconstruction, compressed sampling, transfer learning, and cross-organ generalization research.

## 4.2. fastMRI Data Structure

(1) Covered Organs and Modalities

- Knee MRI: Over ~1,500 fully sampled + 10,000 clinical DICOM images.
- Brain MRI: Approximately 6,970 fully sampled (1.5 T / 3 T) including T1, T2, FLAIR and other sequences.
- Prostate MRI: 312 cases of 3 T acquired axial T2 + DWI sequences.
- Breast MRI: 300 cases of 3 T dynamic contrast-enhanced (DCE) MR, using radial k-space sampling.

(2) Data Format and Annotations

- Provide **raw k-space data** (ISMRMRD or vendor-neutral format) + **reconstructed DICOM/NIfTI images**.
- Have undergone de-identification processing (metadata protection/cleaning), each subset is applied for use according to protocol.
- In terms of annotations: although main focus is reconstruction tasks, fastMRI+ subset has also been derived, containing expert bounding box annotations for knee/brain lesions.

(3) Task Types

- Accelerated MRI reconstruction: Recover high-quality images under small k-space sampling.
- Compressed sampling and reconstruction algorithm benchmarks: Provide standard evaluation metrics (such as PSNR, SSIM) to compare different methods.
- Cross-organ transfer learning, model generalization, and weakly supervised learning (with fastMRI+ annotations)

## 4.3. fastMRI Download Methods

- Official homepage: https://fastmri.med.nyu.edu/
- AWS open data storage: https://registry.opendata.aws/nyu-fastmri/
- GitHub code repository: https://github.com/facebookresearch/fastMRI (including data loaders, baseline models)

**Application Process**:

- Need to agree to "Data Sharing Agreement / Dataset Sharing Agreement"
- Fill in institution information, research purposes, etc.
- Data is limited to "research or teaching purposes" and unauthorized redistribution is prohibited.