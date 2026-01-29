---
title: Appendix C-4 Multimodal Datasets
description: Introduces multimodal datasets
---

# Appendix C-4: Multimodal Datasets

# 1. ADNI (Alzheimer's Disease Neuroimaging)

## 1.1. ADNI Introduction

ADNI is a long-term, multi-center, observational study initiated in 2004 by the National Institute on Aging (NIA) in collaboration with multiple centers in Canada and the United States, aiming to **identify biomarkers for Alzheimer's disease (AD) and its early stages (MCI)** through structural imaging (MRI), functional imaging (PET), biomarkers (CSF, blood), cognitive assessments, and genomic data.  
This dataset has become an important benchmark resource for AD research, imaging-machine learning, and prognostic modeling due to its large scale, long-term follow-up, and multi-modal data sharing policies. 

## 1.2. ADNI Data Structure

(1) Study Phases

ADNI is divided into multiple phases (ADNI-1, ADNI-GO, ADNI-2, ADNI-3, ADNI-4) to expand the population and imaging modalities:

- ADNI-1 (2004–2010): Approximately 200 healthy controls + 400 MCI + 200 AD patients.
- ADNI-GO / ADNI-2 / ADNI-3: Gradually expanded early MCI, late MCI, Tau PET, functional imaging, etc.
- ADNI-4 (from 2022): Aims to improve research generalization ability, including remote sampling, more multi-center data.

(2) Data Types

ADNI provides the following main data types:

- **Imaging Data**: MRI (structural, functional, diffusion), PET (such as Aβ, FDG, Tau)
- **Biomarkers/Biological Samples**: Blood, CSF, genomic/omics data, pathology data
- **Cognitive/Clinical Data**: Including demographics, education years, MMSE, CDR, neuropsychological tests, follow-up status
- **Multimodal Fusion Data**: Such as imaging + cognitive + genetic + biomarkers, used for predicting disease progression.

(3) Data Scale and Access

- The database contains thousands of subjects, tens of thousands of imaging/biological sample measurements.
- Data is provided through the LONI Image & Data Archive (IDA) platform, requiring application submission and data use agreement (DUA) signing to access.
- Researchers can download MRI, PET, genetic, CSF, and other data according to protocols for research and teaching (but must comply with authorization terms).

## 1.3. ADNI Download Methods

- Official website: https://adni.loni.usc.edu/ — lists research introductions, data/sample access entries.
- Data access process: Apply for account → sign data use agreement → after approval, log in to IDA → search projects → download required data.
- User guide/data dictionary: "ADNI Data User Guide" provides detailed variable descriptions, data formats, and access requirements.
- Introduction paper: [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://n.neurology.org/content/74/3/201.short)

# 2. TCIA (The Cancer Imaging Archive)

## 2.1. TCIA Introduction

TCIA is a public imaging data platform supported by the US National Cancer Institute (NCI) and operated by the University of Arkansas for Medical Sciences, focusing on archiving, de-identification, and sharing of tumor imaging (CT, MRI, PET, digital pathology, etc.) data.  
Its goal is to promote radiomics, radiogenomics, tumor-genomics joint research, and AI algorithm development and validation. Data is organized in "collections" (disease type/modality/research topic) form, widely used for tumor detection, segmentation, treatment response evaluation, and imaging-omics analysis.

## 2.2. TCIA Data Structure

(1) Organizational Structure

- Data is organized by **Collections**, where each collection typically revolves around a **tumor type** (such as lung cancer, breast cancer, brain tumor), **imaging modality** (CT, MRI, PET, digital pathology), or **research topic**.
- Each collection usually includes several patients (subjects), each patient with several examinations (studies), each examination with several series/slices.

(2) Imaging Modalities and Additional Data

- Main modalities include: CT, MRI, PET, as well as digital pathology (Whole Slide Images) and structured support data (such as clinical data, genomic data, radiation therapy structures, dose plans, etc.)
- All images are basically in DICOM format.
- Support data: Some collections provide clinical follow-up, treatment information, image annotations (ROI/segmentation), genomic data links, etc.

(3) Scale Indicators

- As of publication time, TCIA contains **over 30.9 million images**, approximately **37,568 subjects**.
- The number of collections is continuously updated, with each collection covering different numbers of subjects, modalities, and annotation richness.

(4) Task Applicability

- Tumor detection/segmentation: Using CT/MRI tumor mass ROI data or lesion annotations.
- Imaging-omics/radiomics: Joint modeling of imaging features + clinical/genetic data.
- Multimodal fusion: Imaging + pathology + gene + treatment response.
- Algorithm validation benchmarks: Due to multi-center, real clinical data sources, suitable for evaluating generalization ability.

## 3.3. TCIA Download Methods

- Official homepage: https://www.cancerimagingarchive.net/