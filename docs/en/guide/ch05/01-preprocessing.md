---
title: 5.1 Preprocessing (Modality-Specific Considerations)
description: Exploring deep learning preprocessing techniques for different medical imaging modalities, including CT windowing, MRI bias correction, and X-ray enhancement
---

# 5.1 Preprocessing (Modality-Specific Considerations)

> "Good preprocessing is the foundation of successful deep learning models—garbage in, garbage out." — The Golden Rule of Medical Image AI

In the previous chapters, we learned about medical imaging principles, reconstruction algorithms, and quality assessment. Now, we enter the deep learning era, facing new challenges: **how to prepare medical image data from different modalities into a format suitable for deep learning model input?**

Unlike natural images, medical images have unique physical characteristics and clinical requirements. CT Hounsfield units, MRI multi-sequence characteristics, X-ray contrast limitations—each modality requires specialized preprocessing strategies. This chapter will delve into these modality-specific preprocessing techniques to lay a solid foundation for subsequent deep learning tasks.

---

## 🏥 Medical Image Preprocessing Importance

### Medical Images vs Natural Images

Medical images differ fundamentally from the natural images we encounter in daily life:

| Characteristic         | Natural Images      | Medical Images                                                     |
| ---------------------- | ------------------- | ------------------------------------------------------------------ |
| **Data Range**         | 0-255 (8-bit)       | Modality-specific (HU values, arbitrary units, etc.)               |
| **Physical Meaning**   | RGB color intensity | Physical measurements (attenuation, magnetization intensity, etc.) |
| **Standardization**    | Relatively standard | Highly dependent on equipment and scanning parameters              |
| **Region of Interest** | Entire image        | Specific tissues or lesions                                        |
| **Prior Knowledge**    | Limited             | Rich anatomical and physiological priors                           |

::: info 🧠 The "Appetite" of Deep Learning
Deep learning models, especially CNNs, typically expect:
- **Standardized input ranges**: such as [0, 1] or [-1, 1]
- **Consistent resolution**: fixed image dimensions
- **Normalized contrast**: avoid certain channels dominating training
- **Noise and artifact removal**: improve model generalization ability

The core task of medical image preprocessing is to convert original physical measurements into a format that models "like."
:::

### Hierarchy of Preprocessing

Medical image preprocessing can be divided into three levels:

![Medical Image Preprocessing Hierarchy](/images/ch05/01-preprocessing-hierarchy-en.png)
*Figure: Three levels of medical image preprocessing from basic to modality-specific to task-oriented.*

<details>
<summary>📖 View Original Mermaid Code</summary>

```mermaid
graph TD
    A[Raw Medical Images] --> B[Basic Preprocessing]
    B --> C[Modality-Specific Preprocessing]
    C --> D[Task-Oriented Preprocessing]

    B --> B1[Resampling]
    B --> B2[Orientation Standardization]
    B --> B3[Size Adjustment]

    C --> C1[CT: Windowing]
    C --> C2[MRI: Bias Correction]
    C --> C3[X-ray: Contrast Enhancement]

    D --> D1[Data Augmentation]
    D --> D2[Normalization]
    D --> D3[Batch Balancing]
```
</details>

---

## 🫧 CT Preprocessing Techniques

### Theoretical Foundation of HU Values

In Chapter 1, we learned the definition of Hounsfield Units (HU):

$$
HU = 1000 \times \frac{\mu_{tissue} - \mu_{water}}{\mu_{water} - \mu_{air}}
$$

This physically meaningful metric gives CT images **absolute comparability**—regardless of which hospital's scanner is used, water's HU value is always 0, and air is always -1000.

### Challenge: Dynamic Range vs Tissue of Interest

**Problem**: CT HU values range from -1000 (air) to +3000+ (dense bone or metal), while deep learning models typically struggle to handle such large dynamic ranges.

**Solution**: Windowing technology

#### Windowing Principles

Windowing maps HU values to display or processing ranges:

$$
I_{output} = \text{clip}\left(\frac{HU - \text{WindowLevel}}{\text{WindowWidth}} \times 255 + 128, 0, 255\right)
$$

Where:
- `WindowLevel`: Center HU value of the window
- `WindowWidth`: HU value range of the window
- `clip()`: Limits output to [0, 255] range

#### Clinically Common Windows

| Window Type            | Window Level | Window Width | Applicable Tissue      | Visible Structures                             |
| ---------------------- | ------------ | ------------ | ---------------------- | ---------------------------------------------- |
| **Lung Window**        | -600         | 1500         | Lung tissue            | Lung markings, small nodules, pneumothorax     |
| **Mediastinal Window** | 50           | 350          | Mediastinal structures | Heart, great vessels, lymph nodes              |
| **Bone Window**        | 300          | 2000         | Bones                  | Cortical bone, bone marrow, microfractures     |
| **Brain Window**       | 40           | 80           | Brain tissue           | Gray matter, white matter, cerebrospinal fluid |
| **Abdominal Window**   | 50           | 400          | Abdominal organs       | Liver, pancreas, kidneys                       |

::: tip 💡 The Art of Window Selection
Window selection is like camera focusing:
- **Narrow window**: High contrast, rich details but limited range
- **Wide window**: Large coverage but reduced contrast
- **Multi-window strategy**: For complex tasks, multiple windows can be used as different input channels
:::

### HU Clipping and Outlier Handling

#### HU Clipping Strategy

```python
def clip_hu_values(image, min_hu=-1000, max_hu=1000):
    """
    HU value clipping: remove extreme values, retain tissue range of interest
    """
    # Deep copy to avoid modifying original data
    processed_image = image.copy()

    # Clip HU values
    processed_image[processed_image < min_hu] = min_hu
    processed_image[processed_image > max_hu] = max_hu

    return processed_image
```

**Common Clipping Ranges:**
- **Soft tissue range**: [-200, 400] HU (exclude air and dense bone)
- **Full body range**: [-1000, 1000] HU (include most clinically relevant structures)
- **Bone tissue range**: [-200, 3000] HU (suitable for bone analysis)

#### Metal Artifact Detection and Processing

Metal implants (such as dental fillings, hip prostheses) can produce extreme HU values and streak artifacts:

```python
def detect_metal_artifacts(image, threshold=3000):
    """
    Detect metal artifact regions
    """
    metal_mask = image > threshold

    # Connectivity analysis, remove isolated noise points
    from scipy import ndimage
    labeled_mask, num_features = ndimage.label(metal_mask)

    # Retain large-area metal regions
    significant_metal = np.zeros_like(metal_mask)
    for i in range(1, num_features + 1):
        if np.sum(labeled_mask == i) > 100:  # Minimum area threshold
            significant_metal[labeled_mask == i] = True

    return significant_metal
```

### Practical Case: Lung Cancer Screening Preprocessing

![CT Lung Nodule Preprocessing Pipeline](https://ars.els-cdn.com/content/image/1-s2.0-S1361841515000035-gr3.jpg)
*CT lung nodule detection preprocessing pipeline: from raw DICOM to model input*

**Complete Preprocessing Pipeline:**
1. **DICOM reading**: Extract pixel data and HU value calibration information
2. **HU value conversion**: Apply rescale slope and intercept
3. **Lung region extraction**: Based on HU value thresholding and connectivity analysis
4. **Resampling**: Unify to isotropic resolution (e.g., 1mm³)
5. **Windowing**: Apply lung window (window level -600, window width 1500)
6. **Normalization**: Map to [0, 1] range
7. **Size adjustment**: Crop or padding to fixed size

---

## 🧲 MRI Preprocessing Techniques

### MRI Intensity Inhomogeneity Problem

#### Causes of Bias Field

MRI signal intensity inhomogeneity (bias field) is a common problem, mainly originating from:

1. **RF field inhomogeneity**: Coil sensitivity variations
2. **Gradient field nonlinearity**: Nonlinear distortion of gradient fields
3. **Tissue property variations**: Different tissue magnetic susceptibility differences
4. **Patient-related factors**: Body shape, respiratory motion, etc.

**Impact of bias field:**
- Same tissue shows different signal intensities at different locations
- Quantitative analysis (such as volume measurement) produces bias
- Deep learning models learn artifact features rather than true anatomical features

#### Bias Field Visualization

```python
def visualize_bias_field(image, corrected_image):
    """
    Visualize bias field correction effect
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Bias field estimation
    bias_field = image / (corrected_image + 1e-6)
    axes[1].imshow(bias_field, cmap='hot')
    axes[1].set_title('Estimated Bias Field')
    axes[1].axis('off')

    # Corrected image
    axes[2].imshow(corrected_image, cmap='gray')
    axes[2].set_title('Corrected Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
```

### N4ITK Bias Field Correction Algorithm

#### N4ITK Algorithm Principle

**N4ITK** (N4 Iterative Bias Correction) is currently the most widely used bias field correction algorithm:

$$
I_{corrected}(x) = \frac{I_{original}(x)}{B(x)} + \epsilon
$$

Where:
- $I_{original}(x)$: Original signal intensity
- $B(x)$: Estimated bias field
- $\epsilon$: Small constant to avoid division by zero

**Algorithm Features:**
- Based on B-spline field bias field modeling
- Iterative optimization process
- Maintains tissue boundary integrity
- Effective for various MRI sequences

#### N4ITK Implementation

```python
import nibabel as nib
import numpy as np
from dipy.denoise import bias_field_correction

class N4ITKBiasCorrector:
    """N4ITK bias field corrector implementation"""

    def __init__(self, max_iterations=50, shrink_factor=2):
        self.max_iterations = max_iterations
        self.shrink_factor = shrink_factor

    def correct_bias_field(self, image, output_path=None):
        """Execute N4ITK bias field correction"""
        # 1. Downsample for faster processing
        working_image = self._downsample(image)

        # 2. Iteratively optimize bias field
        bias_field = self._optimize_bias_field(working_image)

        # 3. Apply correction and restore resolution
        corrected_image = self._apply_correction(image, bias_field)

        return corrected_image
```

### White Stripe Intensity Normalization

#### White Stripe Algorithm Principle

**White Stripe** is a simple yet effective MRI intensity normalization method:

1. **Identify white matter region**: In brain MRI, white matter has relatively stable signal characteristics
2. **Extract white matter intensity range**: Find the dominant mode of white matter through statistical analysis
3. **Linear mapping**: Map white matter range to standard interval (e.g., [0, 1])

[📖 **Complete Code Example**: `white_stripe_normalization/`](https://github.com/datawhalechina/med-imaging-primer/tree/main/src/ch05/) - Full White Stripe normalization implementation with multi-modality support]

**Execution Results Analysis:**

```
White Stripe normalization:
  Modality: T1
  Original statistics: mean=152.3, std=87.6, range=[0, 255]
  White matter peak: intensity=164.2, width=16.4
  White matter range: [147.8, 180.6]
  Normalized statistics: mean=0.50, std=0.15, range=[0.0, 1.0]
```

**Algorithm Analysis:** White Stripe normalization identifies the white matter intensity peak through histogram analysis and uses it as a reference for intensity standardization. The execution results show that the original T1 image has a white matter peak at intensity 164.2 with a width of 16.4. By mapping the white matter range to [0, 1], the algorithm achieves intensity standardization across different scans while preserving tissue contrast relationships.

### Multi-sequence MRI Fusion Strategies

#### Value of Multi-sequence Information

Different MRI sequences provide complementary tissue information:

| Sequence                 | T1-weighted          | T2-weighted       | FLAIR              | DWI               |
| ------------------------ | -------------------- | ----------------- | ------------------ | ----------------- |
| **Tissue Contrast**      | Anatomical structure | Lesion detection  | Lesion boundary    | Cell density      |
| **CSF**                  | Low signal           | High signal       | Low signal         | b-value dependent |
| **White Matter Lesions** | Low contrast         | High contrast     | Very high contrast | Variable          |
| **Acute Infarction**     | Not obvious early    | High signal early | High signal        | Diffusion limited |

![MRI Multi-sequence Comparison](https://www.researchgate.net/publication/349327938/figure/fig2/AS:989495652872194@1614926665094/Different-MRI-sequences-show-the-same-brain-tumor-The-T1-weighted-image-provides.ppm)
*Comparison of different MRI sequences for the same brain tumor, showing complementary information*

#### Multi-sequence Fusion Methods

[📖 **Complete Code Example**: `multisequence_fusion/`](https://github.com/datawhalechina/med-imaging-primer/tree/main/src/ch05/) - Multi-sequence MRI fusion implementation with different strategies]

**Execution Results Analysis:**

```
Multi-sequence fusion:
  Target shape: (128, 128, 128)
  Sequence alignment: T1, T2, FLAIR
  Resampling methods: linear
  Intensity normalization: Z-score
  Fusion strategy: channel stacking
  Output shape: (128, 128, 128, 3)
  Quality metrics: SNR=22.4, CNR=15.8
```

**Algorithm Analysis:** Multi-sequence fusion leverages complementary information from different MRI sequences. The execution results show that three sequences (T1, T2, FLAIR) are successfully aligned to a unified shape of (128, 128, 128) through linear interpolation and normalized using Z-score. The channel stacking strategy creates a 4D tensor with shape (128, 128, 128, 3), preserving both spatial and multi-sequence information. The quality metrics indicate good signal-to-noise ratio (22.4) and contrast-to-noise ratio (15.8), ensuring effective fusion for downstream analysis.

---

## 🦴 X-ray Preprocessing Techniques

### Characteristics and Challenges of X-ray Imaging

#### Inherent Contrast Limitations

The physical principles of X-ray imaging determine its contrast limitations:

1. **Tissue overlap**: 3D structures projected onto 2D plane
2. **Limited dynamic range**: Medical X-ray detectors have relatively limited dynamic range
3. **Scatter influence**: Scattered X-rays reduce image contrast
4. **Exposure parameter variations**: Different examinations use different exposure conditions

### CLAHE Contrast Enhancement

#### CLAHE Algorithm Principle

**CLAHE** (Contrast Limited Adaptive Histogram Equalization) is an improved adaptive histogram equalization:

1. **Block processing**: Divide image into small blocks (e.g., 8×8)
2. **Local histogram equalization**: Perform histogram equalization independently for each block
3. **Contrast limiting**: Limit histogram peaks to avoid noise amplification
4. **Bilinear interpolation**: Use bilinear interpolation at block boundaries for smooth transition

![CLAHE Effect Comparison](https://www.researchgate.net/publication/329926497/figure/fig2/AS:707726086393860@1545445274664/Comparison-of-CHEST-X-RAY-image-enhanced-with-CLAHE.png)
*CLAHE enhancement before and after comparison: left image is original chest X-ray, right image is after CLAHE enhancement*

#### CLAHE Implementation and Optimization

```python
import cv2
import numpy as np

def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE contrast enhancement
    """
    # Ensure input is 8-bit image
    if image.dtype != np.uint8:
        # Normalize and convert to 8-bit
        image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image_8bit = image_normalized.astype(np.uint8)
    else:
        image_8bit = image.copy()

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Apply CLAHE
    enhanced_image = clahe.apply(image_8bit)

    return enhanced_image

def adaptive_clahe_parameters(image):
    """
    Adaptively adjust CLAHE parameters based on image characteristics
    """
    # Calculate image statistical features
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    dynamic_range = np.max(image) - np.min(image)

    # Adaptive parameter adjustment
    if dynamic_range < 50:  # Low contrast image
        clip_limit = 3.0
        tile_size = (16, 16)
    elif mean_intensity < 80:  # Dark image
        clip_limit = 2.5
        tile_size = (12, 12)
    elif mean_intensity > 180:  # Bright image
        clip_limit = 2.0
        tile_size = (8, 8)
    else:  # Normal image
        clip_limit = 2.0
        tile_size = (8, 8)

    return clip_limit, tile_size
```

### Lung Field Segmentation and Normalization

#### Clinical Significance of Lung Field Segmentation

**Lung field segmentation** is a key step in chest X-ray processing:

1. **Region focusing**: Focus processing on lung regions
2. **Background suppression**: Remove interference from regions outside lungs
3. **Standardization**: Standardize lung size and position across different patients
4. **Prior utilization**: Utilize lung anatomical prior knowledge

#### Deep Learning-based Lung Field Segmentation

```python
import torch
import torch.nn as nn

class LungSegmentationNet(nn.Module):
    """
    Simplified lung field segmentation network (U-Net architecture)
    """
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),  # Output binary mask
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def lung_segmentation_preprocessing(image, lung_mask):
    """
    Preprocessing based on lung field segmentation
    """
    # Apply lung field mask
    lung_only = image * lung_mask

    # Calculate statistical parameters of lung region
    lung_pixels = image[lung_mask > 0.5]
    lung_mean = np.mean(lung_pixels)
    lung_std = np.std(lung_pixels)

    # Lung region normalization
    normalized_lungs = (lung_only - lung_mean) / (lung_std + 1e-6)

    # Full image reconstruction (non-lung regions set to 0)
    normalized_image = normalized_lungs

    return normalized_image, (lung_mean, lung_std)
```

---

## 🔧 Common Preprocessing Methods

### Resampling and Resolution Standardization

#### Why Resampling is Needed?

Medical images from different sources may have different spatial resolutions:

| Modality  | Typical Resolution                                | Resolution Variation Reasons                  |
| --------- | ------------------------------------------------- | --------------------------------------------- |
| **CT**    | 0.5-1.0mm (in-plane), 0.5-5.0mm (slice thickness) | Scanning protocols, reconstruction algorithms |
| **MRI**   | 0.5-2.0mm (anisotropic)                           | Sequence types, acquisition parameters        |
| **X-ray** | 0.1-0.2mm (detector size)                         | Magnification, detector type                  |

#### Resampling Methods

```python
import scipy.ndimage as ndimage
import SimpleITK as sitk

def resample_medical_image(image, original_spacing, target_spacing, method='linear'):
    """
    Medical image resampling
    """
    # Calculate scaling factor
    scale_factor = np.array(original_spacing) / np.array(target_spacing)
    new_shape = np.round(np.array(image.shape) * scale_factor).astype(int)

    if method == 'linear':
        # Linear interpolation (suitable for image data)
        resampled_image = ndimage.zoom(image, scale_factor, order=1)
    elif method == 'nearest':
        # Nearest neighbor interpolation (suitable for label data)
        resampled_image = ndimage.zoom(image, scale_factor, order=0)
    elif method == 'bspline':
        # B-spline interpolation (high quality)
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image.SetSpacing(original_spacing)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_shape.tolist())
        resampler.SetInterpolator(sitk.sitkBSpline)

        resampled = resampler.Execute(sitk_image)
        resampled_image = sitk.GetArrayFromImage(resampled)

    return resampled_image
```

### Data Augmentation: Medical-specific Techniques

#### Special Considerations for Medical Image Data Augmentation

Medical image data augmentation needs to consider:

1. **Anatomical reasonableness**: Augmented images must maintain anatomical correctness
2. **Clinical significance**: Augmentation should not alter key pathological features
3. **Modality characteristics**: Different modalities are suitable for different augmentation strategies

#### Spatial Transform Augmentation

```python
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def medical_spatial_augmentation(image, labels=None, augmentation_params=None):
    """
    Medical image spatial transform augmentation
    """
    if augmentation_params is None:
        augmentation_params = {
            'rotation_range': 15,  # degrees
            'scaling_range': 0.1,  # 10%
            'translation_range': 0.05,  # 5%
            'elastic_alpha': 1000,  # Elastic deformation parameters
            'elastic_sigma': 8,
            'enable_augmentation_prob': 0.8
        }

    if np.random.rand() > augmentation_params['enable_augmentation_prob']:
        return image.copy(), labels.copy() if labels is not None else None

    augmented_image = image.copy()
    augmented_labels = labels.copy() if labels is not None else None

    # 1. Random rotation
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-augmentation_params['rotation_range'],
                                 augmentation_params['rotation_range'])
        augmented_image = rotate_3d(augmented_image, angle, axes=(0, 1))
        if augmented_labels is not None:
            augmented_labels = rotate_3d(augmented_labels, angle, axes=(0, 1), order=0)

    # 2. Random scaling
    if np.random.rand() < 0.3:
        scale = np.random.uniform(1 - augmentation_params['scaling_range'],
                                 1 + augmentation_params['scaling_range'])
        augmented_image = zoom_3d(augmented_image, scale)
        if augmented_labels is not None:
            augmented_labels = zoom_3d(augmented_labels, scale, order=0)

    # 3. Elastic deformation (common in medical imaging)
    if np.random.rand() < 0.3:
        augmented_image = elastic_transform_3d(
            augmented_image,
            augmentation_params['elastic_alpha'],
            augmentation_params['elastic_sigma']
        )
        if augmented_labels is not None:
            augmented_labels = elastic_transform_3d(
                augmented_labels,
                augmentation_params['elastic_alpha'],
                augmentation_params['elastic_sigma'],
                order=0
            )

    return augmented_image, augmented_labels

def elastic_transform_3d(image, alpha, sigma, order=1):
    """
    3D elastic deformation - The ace technique for medical image augmentation
    """
    shape = image.shape

    # Generate smooth deformation field
    dx = ndimage.gaussian_filter(np.random.randn(*shape), sigma, mode='constant') * alpha
    dy = ndimage.gaussian_filter(np.random.randn(*shape), sigma, mode='constant') * alpha
    dz = ndimage.gaussian_filter(np.random.randn(*shape), sigma, mode='constant') * alpha

    # Create coordinate grid
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')

    # Apply deformation
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))

    interpolator = RegularGridInterpolator((np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])),
                                          image, method='linear', bounds_error=False, fill_value=0)

    distorted = interpolator(indices).reshape(shape)

    return distorted
```

![Data Augmentation Effects](https://miro.medium.com/v2/resize:fit:1400/1*RjT1_pYfjA3m4WJyAInj6Q.png)
*Medical image data augmentation effects: from left to right are original image, rotation, elastic deformation, brightness adjustment*

---

## 📊 Preprocessing Best Practices

### Preprocessing Workflow Selection Guide

#### Task-driven Preprocessing Strategy

![Task-driven Preprocessing Strategy](/images/ch05/02-preprocessing-strategy-en.png)
*Figure: Decision flow for selecting appropriate preprocessing strategies based on imaging modality.*

<details>
<summary>📖 View Original Mermaid Code</summary>

```mermaid
flowchart TD
    A[Medical Image Preprocessing Task] --> B{Determine Imaging Modality}

    B -->|CT| C[HU Value Calibration]
    C --> D[Window Level Adjustment]
    D --> E[Outlier Processing]

    B -->|MRI| F[Bias Field Correction]
    F --> G[Intensity Standardization]
    G --> H[Multi-sequence Fusion]

    B -->|X-ray| I[Contrast Enhancement]
    I --> J[Anatomical Region Segmentation]
    J --> K[Local Normalization]

    E --> L[Universal Preprocessing]
    H --> L
    K --> L

    L --> M[Resampling]
    M --> N[Size Standardization]
    N --> O[Data Augmentation]
    O --> P[Final Normalization]
```
</details>

### Common Pitfalls and Solutions

#### Preprocessing Pitfalls

| Pitfall Type                 | Specific Manifestation                 | Consequences                             | Solutions                             |
| ---------------------------- | -------------------------------------- | ---------------------------------------- | ------------------------------------- |
| **Over-smoothing**           | Using Gaussian filtering for denoising | Loss of details, small lesions disappear | Use edge-preserving denoising         |
| **Improper normalization**   | Global statistics normalization        | Abnormal regions suppressed              | Local or adaptive normalization       |
| **Information leakage**      | Using test set statistics              | Overly optimistic performance            | Use only training set statistics      |
| **Anatomical discontinuity** | Excessive spatial transforms           | Anatomical structure destruction         | Reasonable transform parameter limits |

#### Validation Strategies

```python
def validate_preprocessing(original_image, processed_image, roi_mask=None):
    """
    Preprocessing effect validation
    """
    from skimage.metrics import structural_similarity as ssim

    validation_results = {}

    # 1. Basic statistical information
    validation_results['original_stats'] = {
        'mean': np.mean(original_image),
        'std': np.std(original_image),
        'min': np.min(original_image),
        'max': np.max(original_image)
    }

    validation_results['processed_stats'] = {
        'mean': np.mean(processed_image),
        'std': np.std(processed_image),
        'min': np.min(processed_image),
        'max': np.max(processed_image)
    }

    # 2. ROI region analysis (if ROI provided)
    if roi_mask is not None:
        original_roi = original_image[roi_mask > 0]
        processed_roi = processed_image[roi_mask > 0]

        validation_results['roi_correlation'] = np.corrcoef(original_roi.flatten(),
                                                          processed_roi.flatten())[0, 1]
        validation_results['roi_ssim'] = ssim(original_roi, processed_roi,
                                            data_range=processed_roi.max() - processed_roi.min())

    # 3. Global similarity
    validation_results['global_correlation'] = np.corrcoef(original_image.flatten(),
                                                          processed_image.flatten())[0, 1]

    return validation_results
```

---

## 🖼️ Algorithm Demonstrations

Below we showcase the practical effects of our implemented preprocessing algorithms on real data. All code examples can be found and run in the [`ch05-code-examples`](https://github.com/datawhalechina/med-imaging-primer/tree/main/src/ch05/) directory.

### MRI Bias Field Visualization and Correction

![MRI Bias Field Visualization](https://raw.githubusercontent.com/datawhalechina/med-imaging-primer/main/src/ch05/visualize_bias_field/output/bias_field_visualization_division.png)
*MRI bias field visualization: left - original image, center - estimated bias field, right - corrected image*

**Bias field correction performance comparison:**
- Gaussian method: MSE=0.0958, PSNR=10.2dB, SSIM=0.368
- Homomorphic method: MSE=0.1984, PSNR=7.0dB, SSIM=0.149
- Polynomial method: MSE=0.0663, PSNR=11.8dB, SSIM=0.545

![Multiple Bias Field Correction Methods Comparison](https://raw.githubusercontent.com/datawhalechina/med-imaging-primer/main/src/ch05/visualize_bias_field/output/bias_field_methods_comparison.png)
*Performance comparison of different bias field correction methods, showing polynomial method performs best in this example*

### White Stripe Intensity Normalization

![White Stripe Normalization Results](https://raw.githubusercontent.com/datawhalechina/med-imaging-primer/main/src/ch05/white_stripe_normalization/output/white_stripe_t1_normalization.png)
*White Stripe intensity normalization: showing original image, normalized result, difference comparison, and statistical analysis*

**Normalization effects for different MRI sequences:**
- T1 sequence: 7 white matter pixels, normalized mean 0.889
- T2 sequence: 6 white matter pixels, normalized mean 0.886
- FLAIR sequence: 10 white matter pixels, normalized mean 0.888

![Multi-modality MRI Normalization Comparison](https://raw.githubusercontent.com/datawhalechina/med-imaging-primer/main/src/ch05/white_stripe_normalization/output/white_stripe_modality_comparison.png)
*White Stripe normalization effects for different MRI sequences, showing intensity distributions and normalization results*

### CLAHE Contrast Enhancement

![CLAHE Parameter Comparison](https://raw.githubusercontent.com/datawhalechina/med-imaging-primer/main/src/ch05/clahe_enhancement/output/clahe_parameter_comparison.png)
*Effects of different CLAHE parameters, showing progressive enhancement from weak to strongest*

**CLAHE enhancement quantitative evaluation:**
- Contrast improvement factor: 1.05
- Dynamic range expansion factor: 1.33
- Information content improvement factor: 1.14
- Edge strength improvement factor: 18.19
- PSNR: 28.05 dB, SSIM: 0.566

![CLAHE Detailed Analysis](https://raw.githubusercontent.com/datawhalechina/med-imaging-primer/main/src/ch05/clahe_enhancement/output/clahe_detailed_analysis.png)
*Detailed CLAHE enhancement analysis, including edge detection, intensity distribution, and enhancement effect evaluation*

### CT HU Value Clipping

![HU Value Clipping Comparison](https://raw.githubusercontent.com/datawhalechina/med-imaging-primer/main/src/ch05/clip_hu_values/output/hu_clipping_软组织范围.png)
*CT HU value clipping: showing soft tissue range (-200, 400 HU) clipping effect*

**Effects of different clipping strategies:**
- Full range [-1000, 1000]: clipping ratio 41.53%, highest information preservation
- Soft tissue range [-200, 400]: clipping ratio 84.13%, suitable for organ analysis
- Bone range [-200, 3000]: clipping ratio 82.91%, suitable for orthopedic applications
- Lung range [-1500, 600]: clipping ratio 1.31%, specialized for lung examination

### Metal Artifact Detection

![Metal Artifact Detection Results](https://raw.githubusercontent.com/datawhalechina/med-imaging-primer/main/src/ch05/detect_metal_artifacts/output/metal_artifact_detection.png)
*CT metal artifact detection: automatic detection of metal regions and artifact severity assessment*

**Detection effects of different thresholds:**
| Threshold (HU) | Detected Regions | Metal Pixels | Ratio | Severity |
| -------------- | ---------------- | ------------ | ----- | -------- |
| 2000           | 2                | 166          | 0.02% | Slight   |
| 3000           | 2                | 165          | 0.02% | Slight   |
| 4000           | 2                | 133          | 0.01% | Slight   |

![Metal Artifact Threshold Comparison](https://raw.githubusercontent.com/datawhalechina/med-imaging-primer/main/src/ch05/detect_metal_artifacts/output/metal_threshold_comparison.png)
*Comparison of metal artifact detection effects for different HU thresholds*

### Practical Application Recommendations

**Choosing appropriate preprocessing strategies:**

1. **Select core algorithms based on modality**
   - CT: HU value clipping + windowing adjustment
   - MRI: Bias field correction + White Stripe normalization
   - X-ray: CLAHE enhancement + local segmentation

2. **Parameter optimization principles**
   - Start conservatively, enhance gradually
   - Use cross-validation to determine optimal parameters
   - Combine quantitative evaluation with visual effects

3. **Quality check key points**
   - Maintain anatomical structure integrity
   - Avoid over-processing or information loss
   - Ensure processing results conform to medical common sense

**Code usage guide:**
Each algorithm has complete documentation and test cases. We recommend:
1. First run synthetic data examples to understand algorithm effects
2. Use your own data for parameter optimization
3. Establish quality check processes to ensure processing effects

---

## 🔑 Key Takeaways

1. **Modality Specificity**: Different imaging modalities require specialized preprocessing strategies
   - CT: Focus on HU value ranges and windowing
   - MRI: Address bias field and intensity normalization
   - X-ray: Focus on contrast enhancement and anatomical region processing

2. **Physical Meaning Preservation**: Preprocessing should not destroy the physical meaning of images
   - Absoluteness of HU values
   - Relativity of MRI signal intensities
   - Equipment dependency of X-ray intensities

3. **Clinical Reasonableness**: Preprocessing results must conform to medical common sense
   - Continuity of anatomical structures
   - Reasonableness of tissue contrast
   - Preservation of pathological features

4. **Data-driven Optimization**: Preprocessing parameters should be adjusted according to specific tasks and datasets
   - Cross-validation to determine optimal parameters
   - Combination of qualitative and quantitative evaluation
   - Consider computational efficiency balance

5. **Quality Assurance**: Establish preprocessing quality inspection mechanisms
   - Automated anomaly detection
   - Expert validation processes
   - Version control and reproducibility

---

::: info 💡 Next Steps
Now you have mastered preprocessing techniques for different modality medical images. In the next section (5.2 Image Segmentation: U-Net and its variants), we will deeply study the core technologies of medical image segmentation, understanding how to convert preprocessed images into precise anatomical structure segmentation results.
:::