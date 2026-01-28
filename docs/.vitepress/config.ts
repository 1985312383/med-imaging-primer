import { defineConfig } from 'vitepress'
import { figure } from '@mdit/plugin-figure'
import { MermaidPlugin } from './theme/plugins/mermaid'

export default defineConfig({
  title: '医学影像处理开源教程',
  description: '从物理成像原理到深度学习的系统性入门指南',
  head: [
    ['link', { rel: 'icon', href: `/med-imaging-primer/favicon.ico` }]
  ],
  // GitHub Pages 部署路径（仓库名）
  base: '/med-imaging-primer/',

  // 路径重写：将 zh/ 目录映射到根路径
  rewrites: {
    'zh/:rest*': ':rest*'
  },

  // 国际化配置
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      title: '医学影像处理开源教程',
      description: '从物理成像原理到深度学习的系统性入门指南',
      themeConfig: {
        logo: '/favicon.ico',
        nav: [
          { text: '首页', link: '/' },
          { text: '教程', link: '/guide/' },
          {
            text: 'GitHub',
            link: 'https://github.com/datawhalechina/med-imaging-primer',
            target: '_blank'
          }
        ],
        sidebar: {
          '/guide/': [
            {
              text: '导览',
              link: '/guide/'
            },
            {
              text: '第1章 医学影像基础',
              collapsed: true,
              items: [
                {
                  text: '1.1 常见成像模态原理',
                  collapsed: false,
                  items: [
                    { text: '1.1.1 CT：光子与衰减', link: '/guide/ch01/01-modalities/01-ct' },
                    { text: '1.1.2 MRI：磁场与自旋', link: '/guide/ch01/01-modalities/02-mri' },
                    { text: '1.1.3 X射线：投影成像', link: '/guide/ch01/01-modalities/03-xray' },
                    { text: '1.1.4 超声(US)：声波反射', link: '/guide/ch01/01-modalities/04-ultrasound' },
                    { text: '1.1.5 PET/SPECT：功能代谢', link: '/guide/ch01/01-modalities/05-pet' }
                  ]
                },
                { text: '1.2 数据格式标准 (DICOM/NIfTI)', link: '/guide/ch01/02-data-formats' },
                { text: '1.3 常用开源工具', link: '/guide/ch01/03-tools' },
                { text: '1.4 图像质量与典型伪影', link: '/guide/ch01/04-artifacts' }
              ]
            },
            {
              text: '第2章 重建前处理：信号校正',
              collapsed: true,
              items: [
                { text: '2.1 CT/X-ray：正弦图与探测器校正', link: '/guide/ch02/01-ct-preprocessing' },
                { text: '2.2 MRI：K空间去卷绕与填充', link: '/guide/ch02/02-mri-preprocessing' },
                { text: '2.3 PET/US：衰减校正与降噪', link: '/guide/ch02/03-pet-us-preprocessing' }
              ]
            },
            {
              text: '第3章 图像重建算法原理',
              collapsed: true,
              items: [
                { text: '3.1 解析重建 (FBP/FFT)', link: '/guide/ch03/01-analytic-recon' },
                { text: '3.2 迭代重建 (SART/OSEM)', link: '/guide/ch03/02-iterative-recon' },
                { text: '3.3 深度学习重建 (AI4Recon) 简介', link: '/guide/ch03/03-dl-recon' }
              ]
            },
            {
              text: '第4章 重建全流程实战 (Case Study)',
              collapsed: true,
              items: [
                { text: '4.1 实验环境搭建 (ODL/TIGRE)', link: '/guide/ch04/01-setup' },
                { text: '4.2 案例一：CT 正弦图回放与重建', link: '/guide/ch04/02-ct-case' },
                { text: '4.3 案例二：MRI K空间成像实验', link: '/guide/ch04/03-mri-case' },
                { text: '4.4 质量评估：PSNR与SSIM实测', link: '/guide/ch04/04-metrics' }
              ]
            },
            {
              text: '第5章 深度学习后处理与新技术',
              collapsed: true,
              items: [
                { text: '5.1 预处理（强调模态差异）', link: '/guide/ch05/01-preprocessing' },
                { text: '5.2 图像分割：U-Net 及其变体', link: '/guide/ch05/02-segmentation' },
                { text: '5.3 分类与检测', link: '/guide/ch05/03-classification' },
                { text: '5.4 图像增强与恢复', link: '/guide/ch05/04-enhancement' },
                { text: '5.5 新范式：大模型(SAM)与生成式AI', link: '/guide/ch05/05-new-tech' }
              ]
            },
            {
              text: '附录',
              collapsed: true,
              items: [
                { text: '附录A：关键公式', link: '/guide/appendix/A-formula' },
                { text: '附录B：工具安装', link: '/guide/appendix/B-tool-Installation' },
                {
                  text: '附录C：公开数据集列表',
                  collapsed: true,
                  items: [
                    { text: 'C.1 CT数据集', link: '/guide/appendix/C-dataset/C-1-CT' },
                    { text: 'C.2 MRI数据集', link: '/guide/appendix/C-dataset/C-2-MRI' },
                    { text: 'C.3 X射线数据集', link: '/guide/appendix/C-dataset/C-3-X-ray' },
                    { text: 'C.4 多模态数据集', link: '/guide/appendix/C-dataset/C-4-Multimodal' }
                  ]
                },
                { text: '附录D：术语表', link: '/guide/appendix/D-glossary' }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/datawhalechina/med-imaging-primer' }
        ],
        footer: {
          message: 'Released under the MIT License.',
          copyright: 'Copyright © 2025-present Your Name'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      title: 'Medical Imaging Primer',
      description: 'An Open Primer on Medical Imaging: From Physics to Deep Learning',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Tutorial', link: '/en/guide/' },
          {
            text: 'GitHub',
            link: 'https://github.com/datawhalechina/med-imaging-primer',
            target: '_blank'
          }
        ],
        sidebar: {
          '/en/guide/': [
            {
              text: 'Introduction',
              link: '/en/guide/'
            },
            {
              text: 'Chapter 1: Medical Imaging Basics',
              collapsed: true,
              items: [
                {
                  text: '1.1 Imaging Modality Principles',
                  collapsed: false,
                  items: [
                    { text: '1.1.1 CT: Photons & Attenuation', link: '/en/guide/ch01/01-modalities/01-ct' },
                    { text: '1.1.2 MRI: Magnetic Field & Spin', link: '/en/guide/ch01/01-modalities/02-mri' },
                    { text: '1.1.3 X-ray: Projection Imaging', link: '/en/guide/ch01/01-modalities/03-xray' },
                    { text: '1.1.4 Ultrasound (US): Acoustic Reflection', link: '/en/guide/ch01/01-modalities/04-ultrasound' },
                    { text: '1.1.5 PET/SPECT: Metabolism & Function', link: '/en/guide/ch01/01-modalities/05-pet' }
                  ]
                },
                { text: '1.2 Data Formats (DICOM/NIfTI)', link: '/en/guide/ch01/02-data-formats' },
                { text: '1.3 Common Open-Source Tools', link: '/en/guide/ch01/03-tools' },
                { text: '1.4 Image Quality & Artifacts', link: '/en/guide/ch01/04-artifacts' }
              ]
            },
            {
              text: 'Chapter 2: Pre-Reconstruction Processing',
              collapsed: true,
              items: [
                { text: '2.1 CT/X-ray: Sinogram & Detector Correction', link: '/en/guide/ch02/01-ct-preprocessing' },
                { text: '2.2 MRI: k-Space Unwrapping & Padding', link: '/en/guide/ch02/02-mri-preprocessing' },
                { text: '2.3 PET/US: Attenuation Correction & Denoising', link: '/en/guide/ch02/03-pet-us-preprocessing' }
              ]
            },
            {
              text: 'Chapter 3: Image Reconstruction Principles',
              collapsed: true,
              items: [
                { text: '3.1 Analytic Reconstruction (FBP/FFT)', link: '/en/guide/ch03/01-analytic-recon' },
                { text: '3.2 Iterative Reconstruction (SART/OSEM)', link: '/en/guide/ch03/02-iterative-recon' },
                { text: '3.3 Deep Learning Reconstruction (AI4Recon)', link: '/en/guide/ch03/03-dl-recon' }
              ]
            },
            {
              text: 'Chapter 4: End-to-End Case Studies',
              collapsed: true,
              items: [
                { text: '4.1 Environment Setup (ODL/TIGRE)', link: '/en/guide/ch04/01-setup' },
                { text: '4.2 Case 1: CT Sinogram Playback & Reconstruction', link: '/en/guide/ch04/02-ct-case' },
                { text: '4.3 Case 2: MRI k-Space Imaging Lab', link: '/en/guide/ch04/03-mri-case' },
                { text: '4.4 Metrics: PSNR & SSIM', link: '/en/guide/ch04/04-metrics' }
              ]
            },
            {
              text: 'Chapter 5: Deep Learning Post-Processing & New Trends',
              collapsed: true,
              items: [
                { text: '5.1 Preprocessing (Modality-Specific)', link: '/en/guide/ch05/01-preprocessing' },
                { text: '5.2 Image Segmentation: U-Net and its Variants', link: '/en/guide/ch05/02-segmentation' },
                { text: '5.3 Classification and Detection', link: '/en/guide/ch05/03-classification' },
                { text: '5.4 Image Enhancement and Restoration', link: '/en/guide/ch05/04-enhancement' },
                { text: '5.5 New Paradigms: SAM & Generative AI', link: '/en/guide/ch05/05-new-tech' }
              ]
            },
            {
              text: 'Appendix',
              collapsed: true,
              items: [
                { text: 'Appendix A: Key Formulas', link: '/en/guide/appendix/A-formula' },
                { text: 'Appendix B: Tool Installation', link: '/en/guide/appendix/B-tool-Installation' },
                {
                  text: 'Appendix C: Public Datasets',
                  collapsed: true,
                  items: [
                    { text: 'C.1 CT Datasets', link: '/en/guide/appendix/C-dataset/C-1-CT' },
                    { text: 'C.2 MRI Datasets', link: '/en/guide/appendix/C-dataset/C-2-MRI' },
                    { text: 'C.3 X-ray Datasets', link: '/en/guide/appendix/C-dataset/C-3-X-ray' },
                    { text: 'C.4 Multimodal Datasets', link: '/en/guide/appendix/C-dataset/C-4-Multimodal' }
                  ]
                },
                { text: 'Appendix D: Glossary', link: '/en/guide/appendix/D-glossary' }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/datawhalechina/med-imaging-primer' }
        ],
        footer: {
          message: 'Released under the MIT License.',
          copyright: 'Copyright © 2025-present Your Name'
        }
      }
    }
  },

  // Markdown 扩展
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true,
    config: (md) => {
      md.use(figure)
      md.use(MermaidPlugin)
    },
    math: true
  },
  themeConfig: {
    search: {
      provider: 'local'
    }
  }
})

