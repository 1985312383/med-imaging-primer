---
title: "1.1.4 Ultrasound (US): Sound Waves & Echoes"
description: History, imaging principles, and key technology evolution of ultrasound
---

# 1.1.4 Ultrasound (US): Sound Waves & Echoes

> “Real-time imaging turned diagnosis into something you can *watch*.” — Why ultrasound is everywhere in clinical practice

After CT, MRI, and X-ray (the “anatomical imaging trio”), ultrasound stands out as the **real-time**, **radiation-free**, **portable**, and **cost-effective** modality that clinicians reach for first in many scenarios.

---

## 🔊 From SONAR to medical imaging

### Early roots

Ultrasound (frequency > 20 kHz) was first widely used in **SONAR** for submarine detection in World War I. Later, it became a workhorse in industrial nondestructive testing.

Bringing it into medicine was harder: biological tissues are complex, and clinicians need **fast** and **repeatable** imaging.

### Clinical pioneers

- **1942**: Karl Dussik attempted ultrasound transmission imaging for brain tumors (limited by the skull).
- **1958**: Ian Donald demonstrated obstetric ultrasound and helped establish medical ultrasound as a clinical tool.

---

## 📡 Imaging principle: echoes + piezoelectric transducers

### Frequency vs penetration

Medical ultrasound typically operates at **1–20 MHz**:

- **Low frequency (1–5 MHz)**: deeper penetration (abdomen)
- **High frequency (7–20 MHz)**: higher resolution (thyroid, vessels)

### Piezoelectric effect

The transducer uses piezoelectric crystals (e.g., PZT):

- Apply voltage → crystal vibrates → emits ultrasound
- Echo returns → crystal vibrates → generates voltage (signal)

![Ultrasound Probe](/images/ch01/ultrasound-probe.jpg)
*A linear-array ultrasound probe with multiple piezoelectric elements*

### Interaction with tissue

- **Reflection**: the main information source; depends on **acoustic impedance mismatch**
- **Scattering**: creates speckle/texture
- **Attenuation**: increases with depth and frequency
- **Refraction**: may cause artifacts

### Common modes

| Mode | What it shows | Typical use |
|------|---------------|-------------|
| **B-mode** | 2D grayscale | general imaging |
| **M-mode** | motion over time | cardiac valves |
| **Color Doppler** | flow overlay | vascular / cardiac |

### Doppler equation

Blood flow velocity can be estimated by:

$$
v = \frac{c \cdot \Delta f}{2 f_0 \cos\theta}
$$

:::: tip 💡 Why gel is required
Air has a very different acoustic impedance than tissue, causing near-total reflection at the air–skin interface. The coupling gel removes air gaps so ultrasound energy can enter the body.
::::

---

## 🚀 Technology evolution (high-level)

| Era | Key breakthroughs | Typical impact |
|-----|-------------------|----------------|
| 1970s–1980s | Real-time B-mode | true dynamic imaging |
| 1980s–1990s | Doppler | flow/hemodynamics |
| 1990s–2000s | 3D/4D, harmonic imaging | better visualization & contrast |
| 2000s–2010s | Contrast US, elastography | functional assessment |
| 2010s–today | AI + handheld devices | accessibility & workflow |

---

## Where to go next

- PET/SPECT: `1.1.5 PET/SPECT: Metabolism & Function` (`/en/guide/ch01/01-modalities/05-pet`)
- Practical preprocessing topics (attenuation correction / denoising): Chapter 2.3 (`/en/guide/ch02/03-pet-us-preprocessing`)


