# Medical Imaging Primer

<div align="center">

![banner](/docs/public/images/banner.png)

**English | [简体中文](README.md)**

</div>

An open-source tutorial on medical imaging processing, providing a systematic guide from physical imaging principles, reconstruction algorithms to deep learning post-processing.

> [!IMPORTANT]
> **Content Source and Copyright Notice**
>
> All images and content in this tutorial are sourced from publicly available internet resources or published academic materials. This tutorial does not contain any trade secrets from medical device manufacturers, nor does it include any unpublished proprietary methods or technologies. If there are any copyright concerns, please feel free to contact us for removal.

> [!NOTE]
> **Intended Audience and Value**
>
> This tutorial aims to provide value to three groups of readers:
> - **Medical Students**: We hope to help you understand the physical principles and technical foundations of medical imaging, though this tutorial does not cover clinical diagnosis.
> - **Biomedical Engineering Students**: We attempt to provide systematic knowledge from physical principles to algorithm implementation.
> - **Computer Science/AI Researchers**: We hope this resource may help you understand the characteristics and preprocessing methods of medical imaging data.
>
> Please note: This is a learning resource and cannot replace systematic professional education.

> [!TIP]
> **Update Timeliness and Accuracy**
>
> For the most up-to-date and accurate content, please refer to the [Chinese version (简体中文)](README.zh.md) as the primary reference.

## 📖 About

This tutorial aims to help beginners systematically learn medical imaging processing technology. No medical or imaging background is required - we start from the basics of CT/MRI and gradually progress to advanced topics including reconstruction algorithms and AI-powered post-processing.

## 🌐 Online Access

<div align="center">

[![Online Documentation](https://img.shields.io/badge/📖_Online_Documentation-Visit_Now-blue?style=for-the-badge)](https://datawhalechina.github.io/med-imaging-primer/)

**👉 [Click here to access the online tutorial](https://datawhalechina.github.io/med-imaging-primer/) 👈**

</div>

## ✨ Features

- **From Scratch**: No prerequisites required, starting from fundamental concepts
- **Principle-First**: Deep dive into physical mechanisms and mathematical models, beyond just API usage
- **Multi-Modal Coverage**: Comprehensive coverage of CT, MRI, X-ray, PET, and Ultrasound imaging
- **Comprehensive**: Covers the complete pipeline from raw data → reconstruction → AI post-processing
- **Open Source**: Continuously updated content, contributions and discussions welcome

## 📚 Content Outline

### ✅ Completed Chapters

#### Chapter 1: Medical Imaging Basics
- **1.1** Common Imaging Modality Principles — Understanding CT, MRI, X-ray, PET, and Ultrasound
- **1.2** Data Format Standards — Mastering DICOM, NIfTI, and format conversion
- **1.3** Common Open Source Tools — Practical guide to ITK, SimpleITK, and visualization tools
- **1.4** Artifacts in Medical Imaging — Identifying and understanding imaging artifacts

#### Chapter 2: Modality-Specific Preprocessing
- **2.1** CT: From Detector Signal to Corrected Projection
  - Complete workflow from photon acquisition to calibrated projection data
- **2.2** MRI k-space Preprocessing ✨ **NEW**
  - K-space data generation, undersampling strategies, and filtering techniques

#### Chapter 3: Image Reconstruction Algorithms
- **3.1** Analytic Reconstruction: Filtered Back Projection (FBP)
  - Mathematical principles of CT reconstruction and filter selection
- **3.2** Iterative Reconstruction: Algebraic and Compressed Sensing Methods ✨ **NEW**
  - POCS, SART and other iterative algorithms: principles and implementation

#### Chapter 4: Multi-Modality Reconstruction Practice
- **4.1** Dataset Introduction and Experiment Preparation
- **4.2** Case Study 1: Complete CT Reconstruction Workflow ✨ **NEW**
  - Full pipeline from raw projection data to reconstructed images
- **4.3** Case Study 2: MRI K-space Imaging and Reconstruction Experiments ✨ **NEW**
  - Undersampling strategies, reconstruction algorithms, and quality assessment

#### Chapter 5: Deep Learning Post-processing
- **5.1** Preprocessing — Modality-specific data preparation for AI models
- **5.2** Image Segmentation：U-Net and its Variants  — Semantic segmentation of medical structures
- **5.3** Classification and Detection — Automated diagnosis and lesion detection
- **5.4** Image Enhancement and Recovery — Denoising and super-resolution techniques

### 🚀 Coming Soon
- **2.3** X-ray Direct Imaging Correction
- **4.4** More Reconstruction Cases (PET, Ultrasound, etc.)
- **Chapter 6** Advanced Topics and Cutting-Edge Techniques

## 🤝 Contributing

We welcome contributions of all kinds! Whether it's:

- 📝 Improving documentation and fixing typos
- 🌐 Adding translations
- 💡 Suggesting new content or topics
- 🐛 Reporting issues
- 📖 Writing new chapters or sections

### How to Contribute

1. Fork this repository
2. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
3. Push to the branch (`git push origin feature/AmazingFeature`)
4. Open a Pull Request

### Content Guidelines

- Use clear and accessible language
- Include practical examples where possible
- Provide references to academic papers or authoritative sources
- Follow the existing structure and formatting style

## 👥 Contributors

Thanks goes to these wonderful people:

<div align="center">

<a href="https://github.com/datawhalechina/med-imaging-primer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=datawhalechina/med-imaging-primer" />
</a>

</div>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Quick Links

- **📖 Online Documentation**: [https://datawhalechina.github.io/med-imaging-primer/](https://datawhalechina.github.io/med-imaging-primer/)
- **💻 GitHub Repository**: [https://github.com/datawhalechina/med-imaging-primer](https://github.com/datawhalechina/med-imaging-primer)

## 💬 Feedback & Participation

We value your feedback and encourage community participation:

- **🐛 Report Issues**: Found a bug or error? [Open an Issue](https://github.com/datawhalechina/med-imaging-primer/issues)
- **💡 Suggest Ideas**: Have suggestions for new content? [Start a Discussion](https://github.com/datawhalechina/med-imaging-primer/discussions)
- **🤝 Contribute**: Want to contribute? See the [Contributing](#-contributing) section above

---

⭐ If you find this tutorial helpful, please consider giving it a star!

