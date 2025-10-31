import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Medical Imaging Primer',
  description: 'An Open Primer on Medical Imaging: From Physics to Deep Learning',

  // GitHub Pages 部署路径（仓库名）
  base: '/med-imaging-primer/',

  // 路径重写：将 en/ 目录映射到根路径
  rewrites: {
    'en/:rest*': ':rest*'
  },

  // 国际化配置
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      title: 'Medical Imaging Primer',
      description: 'An Open Primer on Medical Imaging: From Physics to Deep Learning',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/' },
          { text: 'Tutorial', link: '/guide/' },
          {
            text: 'GitHub',
            link: 'https://github.com/1985312383/med-imaging-primer',
            target: '_blank'
          }
        ],
        sidebar: {
          '/guide/': [
            {
              text: 'Introduction',
              link: '/guide/'
            },
            {
              text: 'Chapter 1: Medical Imaging Basics',
              collapsed: true,
              items: [
                { text: '1.1 Common Imaging Modality Principles', link: '/guide/ch01/01-modalities' },
                { text: '1.2 Data Format Standards', link: '/guide/ch01/02-data-formats' },
                { text: '1.3 Common Open-Source Tools', link: '/guide/ch01/03-tools' },
                { text: '1.4 Image Quality and Typical Artifacts', link: '/guide/ch01/04-artifacts' }
              ]
            },
            {
              text: 'Chapter 2: Pre-Reconstruction Processing',
              collapsed: true,
              items: [
                { text: '2.1 CT: From Detector Signal to Corrected Projection', link: '/guide/ch02/01-ct-preprocessing' },
                { text: '2.2 MRI: k-Space Data Preprocessing', link: '/guide/ch02/02-mri-preprocessing' },
                { text: '2.3 X-ray: Direct Imaging Correction', link: '/guide/ch02/03-xray-preprocessing' }
              ]
            },
            {
              text: 'Chapter 3: Image Reconstruction Algorithms',
              collapsed: true,
              items: [
                { text: '3.1 CT Reconstruction', link: '/guide/ch03/01-ct-reconstruction' },
                { text: '3.2 MRI Reconstruction', link: '/guide/ch03/02-mri-reconstruction' },
                { text: '3.3 X-ray Imaging', link: '/guide/ch03/03-xray-imaging' }
              ]
            },
            {
              text: 'Chapter 4: Reconstruction Practice and Validation',
              collapsed: true,
              items: [
                { text: '4.1 CT Complete Workflow', link: '/guide/ch04/01-ct-workflow' },
                { text: '4.2 MRI Reconstruction Example', link: '/guide/ch04/02-mri-example' },
                { text: '4.3 X-ray Correction Example', link: '/guide/ch04/03-xray-example' },
                { text: '4.4 Reconstruction Quality Assessment', link: '/guide/ch04/04-quality-assessment' },
                { text: '4.5 Common Issues Troubleshooting Guide', link: '/guide/ch04/05-troubleshooting' }
              ]
            },
            {
              text: 'Chapter 5: Medical Image Post-Processing',
              collapsed: true,
              items: [
                { text: '5.1 Preprocessing (Modality-Specific)', link: '/guide/ch05/01-preprocessing' },
                { text: '5.2 Image Segmentation', link: '/guide/ch05/02-segmentation' },
                { text: '5.3 Classification and Detection', link: '/guide/ch05/03-classification' },
                { text: '5.4 Image Enhancement and Restoration', link: '/guide/ch05/04-enhancement' }
              ]
            },
            {
              text: 'Appendix',
              collapsed: true,
              items: [
                { text: 'Appendix A: Key Formulas', link: '/guide/appendix/a-formulas' },
                { text: 'Appendix B: Tool Installation', link: '/guide/appendix/b-installation' },
                { text: 'Appendix C: Public Datasets', link: '/guide/appendix/c-datasets' },
                { text: 'Appendix D: Glossary', link: '/guide/appendix/d-glossary' }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/1985312383/med-imaging-primer' }
        ],
        footer: {
          message: 'Released under the MIT License.',
          copyright: 'Copyright © 2025-present Your Name'
        }
      }
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      title: '医学影像处理开源教程',
      description: '从物理成像原理到深度学习的系统性入门指南',
      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '教程', link: '/zh/guide/' },
          {
            text: 'GitHub',
            link: 'https://github.com/1985312383/med-imaging-primer',
            target: '_blank'
          }
        ],
        sidebar: {
          '/zh/guide/': [
            {
              text: '导览',
              link: '/zh/guide/'
            },
            {
              text: '第1章 医学影像基础',
              collapsed: true,
              items: [
                { text: '1.1 常见成像模态原理与特点', link: '/zh/guide/ch01/01-modalities' },
                { text: '1.2 数据格式标准', link: '/zh/guide/ch01/02-data-formats' },
                { text: '1.3 常用开源工具', link: '/zh/guide/ch01/03-tools' },
                { text: '1.4 图像质量与典型伪影', link: '/zh/guide/ch01/04-artifacts' }
              ]
            },
            {
              text: '第2章 重建前处理：模态特异性校正流程',
              collapsed: true,
              items: [
                { text: '2.1 CT：从探测器信号到校正投影', link: '/zh/guide/ch02/01-ct-preprocessing' },
                { text: '2.2 MRI：k空间数据预处理', link: '/zh/guide/ch02/02-mri-preprocessing' },
                { text: '2.3 X-ray：直接成像的校正', link: '/zh/guide/ch02/03-xray-preprocessing' }
              ]
            },
            {
              text: '第3章 图像重建算法（按模态组织）',
              collapsed: true,
              items: [
                { text: '3.1 CT重建', link: '/zh/guide/ch03/01-ct-reconstruction' },
                { text: '3.2 MRI重建', link: '/zh/guide/ch03/02-mri-reconstruction' },
                { text: '3.3 X-ray成像', link: '/zh/guide/ch03/03-xray-imaging' }
              ]
            },
            {
              text: '第4章 重建实践与验证（多模态示例）',
              collapsed: true,
              items: [
                { text: '4.1 CT完整流程', link: '/zh/guide/ch04/01-ct-workflow' },
                { text: '4.2 MRI重建示例', link: '/zh/guide/ch04/02-mri-example' },
                { text: '4.3 X-ray校正示例', link: '/zh/guide/ch04/03-xray-example' },
                { text: '4.4 重建质量评估', link: '/zh/guide/ch04/04-quality-assessment' },
                { text: '4.5 常见问题排查指南', link: '/zh/guide/ch04/05-troubleshooting' }
              ]
            },
            {
              text: '第5章 医学图像后处理（通用+模态适配）',
              collapsed: true,
              items: [
                { text: '5.1 预处理（强调模态差异）', link: '/zh/guide/ch05/01-preprocessing' },
                { text: '5.2 图像分割', link: '/zh/guide/ch05/02-segmentation' },
                { text: '5.3 分类与检测', link: '/zh/guide/ch05/03-classification' },
                { text: '5.4 图像增强与恢复', link: '/zh/guide/ch05/04-enhancement' }
              ]
            },
            {
              text: '附录',
              collapsed: true,
              items: [
                { text: '附录A：关键公式', link: '/zh/guide/appendix/a-formulas' },
                { text: '附录B：工具安装', link: '/zh/guide/appendix/b-installation' },
                { text: '附录C：公开数据集列表', link: '/zh/guide/appendix/c-datasets' },
                { text: '附录D：术语表', link: '/zh/guide/appendix/d-glossary' }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/1985312383/med-imaging-primer' }
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
    lineNumbers: true
  }
})