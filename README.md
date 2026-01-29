# 医学影像处理开源教程（⚠️ Alpha内测版）

<div align="center">

![banner](/docs/public/images/banner.png)

**[English](README_en.md) | 简体中文**

</div>

一个关于医学影像处理的开源教程，提供从物理成像原理、重建算法到深度学习后处理的系统性指南。

> [!CAUTION]
> ⚠️ Alpha内测版本警告：此为早期内部构建版本，尚不完整且可能存在错误，欢迎大家提Issue反馈问题或建议。

> [!IMPORTANT]
> **内容来源和版权声明**
>
> 本教程中使用的所有图片和内容均来自互联网公开资源或已出版的学术资料。本教程不涉及任何医疗器械公司的商业机密，也不包含任何未公开发表的专有方法或技术。如有版权问题，欢迎联系删除。

> [!NOTE]
> **教程受众和价值说明**
>
> 本教程希望为以下三类读者提供帮助：
> - **医学生**：我们希望能够帮助你理解医学影像的物理原理和技术基础，但本教程不涉及临床诊断。
> - **生物医学工程专业学生**：我们尝试提供从物理原理到算法实现的系统性知识。
> - **计算机科学/AI 领域的研究者**：我们希望这个资源可能有所帮助，让你理解医学影像数据的特性和预处理方法。
>
> 请注意：本教程是一个学习资源，不能替代系统的专业教育。

## 📖 关于本项目

本教程旨在帮助初学者系统性地学习医学影像处理技术。无需医学或影像学背景，我们从 CT/MRI 的基础知识开始，逐步深入到重建算法和 AI 驱动的后处理等高级主题。

## 🌐 在线访问

<div align="center">

[![在线文档](https://img.shields.io/badge/📖_在线文档-立即访问-blue?style=for-the-badge)](https://datawhalechina.github.io/med-imaging-primer/)

**👉 [点击这里访问在线教程](https://datawhalechina.github.io/med-imaging-primer/) 👈**

</div>

## ✨ 项目特点

- **零基础入门**：无需任何前置知识，从基础概念开始讲解
- **原理优先**：深入探讨物理机制和数学模型，而不仅仅是 API 调用
- **多模态覆盖**：全面涵盖 CT、MRI、X-ray、PET 和超声成像
- **内容全面**：覆盖从原始数据 → 重建 → AI 后处理的完整流程
- **开源协作**：内容持续更新，欢迎贡献和讨论

## 📚 内容大纲

### ✅ 已完成章节

#### 第1章 医学影像基础
- **1.1** 常见成像模态原理 — 理解 CT、MRI、X-ray、PET 和超声成像
- **1.2** 数据格式标准 — 掌握 DICOM、NIfTI 及格式转换
- **1.3** 常用开源工具 — ITK、SimpleITK 和可视化工具实用指南
- **1.4** 医学影像中的伪影 — 识别和理解成像伪影

#### 第2章 模态特定的预处理
- **2.1** CT：从探测器信号到校正投影
  - 从光子采集到校准投影数据的完整工作流程
- **2.2** MRI k-空间预处理 ✨ **新增**
  - K空间数据生成、欠采样策略与滤波技术

#### 第3章 图像重建算法
- **3.1** 解析重建：滤波反投影（FBP）
  - CT重建的数学原理与滤波器选择
- **3.2** 迭代重建：代数与压缩感知方法 ✨ **新增**
  - POCS、SART等迭代算法原理与实现

#### 第4章 多模态重建实践
- **4.1** 数据集介绍与实验准备
- **4.2** 案例一：CT重建全流程 ✨ **新增**
  - 从原始投影数据到重建图像的完整流程
- **4.3** 案例二：MRI K空间成像与重建实验 ✨ **新增**
  - 欠采样策略、重建算法与质量评估

#### 第5章 基于深度学习的后处理
- **5.1** 预处理 — 针对 AI 模型的模态特定数据准备
- **5.2** 图像分割：U-Net 及其变种 — 医学结构的语义分割
- **5.3** 分类与检测 — 自动诊断和病灶检测
- **5.4** 图像增强与恢复 — 去噪和超分辨率技术

### 🚀 即将推出
- **2.3** X-ray 直接成像校正
- **4.4** 更多重建案例（PET、超声等）
- **第6章** 高级主题与前沿技术

## 🤝 如何贡献

我们欢迎各种形式的贡献！包括但不限于：

- 📝 改进文档和修正错别字
- 🌐 添加翻译
- 💡 建议新的内容或主题
- 🐛 报告问题
- 📖 编写新的章节或小节

### 贡献步骤

1. Fork 本仓库
2. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
3. 推送到分支 (`git push origin feature/AmazingFeature`)
4. 开启一个 Pull Request

### 内容指南

- 使用清晰易懂的语言
- 尽可能包含实际示例
- 提供学术论文或权威来源的参考文献
- 遵循现有的结构和格式风格

## 👥 贡献者

感谢所有为本项目做出贡献的人：

<div align="center">

<a href="https://github.com/datawhalechina/med-imaging-primer/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=datawhalechina/med-imaging-primer" />
</a>

</div>

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🔗 快速链接

- **📖 在线文档**：[https://datawhalechina.github.io/med-imaging-primer/](https://datawhalechina.github.io/med-imaging-primer/)
- **💻 GitHub 仓库**：[https://github.com/datawhalechina/med-imaging-primer](https://github.com/datawhalechina/med-imaging-primer)

## � 反馈与参与

我们重视您的反馈并鼓励社区参与：

- **🐛 报告问题**：发现错误或 bug？[提交 Issue](https://github.com/datawhalechina/med-imaging-primer/issues)
- **💡 建议想法**：有新内容建议？[发起 Discussion](https://github.com/datawhalechina/med-imaging-primer/discussions)
- **🤝 贡献代码**：想要贡献？请查看上方[如何贡献](#-如何贡献)部分

---

⭐ 如果你觉得这个教程有帮助，请考虑给它一个 star！

