---
title: Appendix C-3: X-ray Type Datasets
description: Introduces X-ray type datasets
---

# Appendix C-3: X-ray Type Datasets

# 3. CheXpert (Large-scale Chest X-ray Lesion Detection Dataset)

## 3.1. CheXpert Introduction

CheXpert is a public chest X-ray image dataset released by Stanford University Medical Imaging and AI Team in 2019, mainly used for multi-label prediction of chest diseases (such as pneumonia, pleural effusion, cardiomegaly, etc.). The dataset is collected from real clinical scenarios, emphasizing the handling of uncertain (uncertainty) labels, and is widely used as a benchmark for chest X-ray AI research.

## 3.2. CheXpert Data Structure

- Contains **224,316 chest X-ray images**, from **65,240 patients**, with time range approximately from October 2002 to July 2017.
- Each image is accompanied by corresponding radiology reports, and **14 observations** are automatically extracted from the reports, with labels divided into three categories: "positive (1)", "negative (0)", and "uncertain (-1/u)".
- The 14 observations include: atelectasis (肺不张), cardiomegaly (心脏扩大), consolidation (实变), edema (水肿), enlarged cardiomediastinum, fracture, lung lesion, lung opacity, pleural effusion, pleural other, pneumonia, pneumothorax, support devices, no finding.
- Data perspective: When there are multiple views (such as frontal + lateral), the model usually takes the maximum value of each view's prediction as the indicator.
- Data division: Official provides training set + validation set; the test set consists of 500 independent studies, annotated by five certified radiologists as reference standard.

## 3.3. CheXpert Download Methods

- Official homepage: https://stanfordmlgroup.github.io/competitions/chexpert/
- Access conditions: Usually need to register an account and agree to the Research Use Agreement (RUA) before downloading data.
- Kaggle mirror (such as "CheXpert-v1.0-small") can also be accessed as a subset. https://www.kaggle.com/datasets/ashery/chexpert

# 2. MIMIC-CXR (Large-scale Chest X-ray Public Dataset)

## 2.1 MIMIC-CXR Introduction

MIMIC-CXR is a **de-identified chest X-ray (chest radiograph) dataset** collected by Beth Israel Deaconess Medical Center (BIDMC) in Boston and organized and made public by MIT Laboratory for Computational Physiology and other units. It contains hundreds of thousands of real clinical chest X-ray images matched with radiology reports, oriented toward image understanding, natural language processing, and decision support research.  
For example, its first version is described as: covering approximately 65,379 patients from 2011-2016 period, 227,835 imaging examinations, and 377,110 images.  
This dataset is considered an important benchmark for chest X-ray AI research due to its large scale, clear structure, and accompanying reports.

## 2.2 MIMIC-CXR Data Structure

- **Image Quantity**: Approximately 377,110 chest X-rays, associated with approximately 227,835 imaging examinations.
- **Patient Quantity**: Approximately 65,379 people.
- **View Types**: Most examinations include frontal and lateral views.
- **Image Format**: Provides DICOM format original images (de-identified) and corresponding report text.
- **Report Text**: Each examination is accompanied by free-text reports written by radiologists, describing imaging findings
- **Data Annotation/Derivation**: Users can extract structured labels (such as lesion presence, device location, etc.) based on report text for classification tasks.
- **Task Types**: Including chest X-ray abnormal detection/classification, radiology report-image pairing, image-text joint modeling.
- **Usage Agreement**: Data has undergone de-identification processing, complying with HIPAA Safe Harbour requirements.

## 2.3 MIMIC-CXR Download Methods

- Official hosting platform: https://physionet.org/content/mimic-cxr/2.1.0/
- Download process usually includes: register account → sign data use agreement (Data Use Agreement, DUA) → approval → download.

# 3. NIH ChestX-ray14 (Chest X-ray Multi-label Public Dataset)

## 3.1 Introduction

NIH ChestX-ray14 is a public chest X-ray image dataset released by the US National Institutes of Health (NIH) Clinical Center, initially released in 2017 under the name "ChestX-ray8", and later expanded to include 14 types of common chest lesions (ChestX-ray14). This dataset contains over 100,000 clinical chest X-rays, accompanied by multi-labels automatically mined from text, and is widely used for chest X-ray classification, detection, and weakly supervised learning research.

## 3.2 Data Structure

- Image count: Approximately **112,120 frontal chest X-rays**, from **30,805 unique patients**.
- Labels: Each image is accompanied by up to 14 chest lesion labels + "No Finding" category; labels are extracted from radiology reports using NLP.
- Multi-label task: Each image may contain multiple lesions simultaneously (for example, edema + lung infiltration + cardiomegaly) → belonging to multi-label classification situation.
- Image format: PNG format (some DICOM versions are accessible in Google Cloud)
- Data division: Training set 86,524 images, test set 25,596 images.
- Common tasks: Chest disease classification, weakly supervised localization (few annotated bounding boxes), multi-label metrics (ROC-AUC) evaluation.

## 3.3 Download Methods

- Official download page: Box link provided by NIH Clinical Center https://nihcc.app.box.com/v/ChestXray-NIHCC
- Google Cloud public storage bucket: https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest?hl=zh-cn
- Kaggle mirror version: Such as "NIH-Chest-X-rays" is provided on Kaggle. https://www.kaggle.com/datasets/nih-chest-xrays/data
- Usage notes: No payment required, no obvious usage restrictions, but requires indicating data source and citing the original paper.