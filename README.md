# Panoramic Affordance Prediction (PAP)

<!-- **[Paper]** | **[Project Page]** | **[Dataset]** | **[Code]** -->

Official repository for the paper: **Panoramic Affordance Prediction**.

> Affordance prediction serves as a critical bridge between perception and action in embodied AI. However, existing research is confined to pinhole camera models, which suffer from narrow Fields of View (FoV) and fragmented observations. In this paper, we present the first exploration into **Panoramic Affordance Prediction**, utilizing 360-degree imagery to capture global spatial relationships and holistic scene understanding. 

<div align="center">
    <img src="assets/teaser.svg" alt="teaser" width="90%">
</div>

## 🌟 Highlights
- **New Task:** We introduce the first exploration into **Panoramic Affordance Prediction**, overcoming the "tunnel vision" of traditional pinhole camera models.
- **PAP-12K Dataset:** A large-scale benchmark featuring 1,003 ultra-high-resolution (12K) panoramic images and over 13,000 carefully annotated reasoning-based QA pairs with pixel-level masks.
- **PAP Framework:** A training-free, coarse-to-fine pipeline mimicking human foveal vision to handle panoramic challenges like geometric distortion, scale variations, and boundary discontinuity.

---

## 🛠️ Method Overview

Our proposed PAP framework operates in three primary stages to tackle 360-degree scenes:

1. **Recursive Visual Routing:** Uses numerical grid prompting to guide Vision-Language Models (VLMs) to dynamically "zoom in" and coarsely locate target tools.
2. **Adaptive Gaze:** Projects the spherical region onto a tailored perspective plane to act as a domain adapter, eliminating geometric distortions and boundary discontinuities.
3. **Cascaded Affordance Grounding:** Deploys robust 2D vision models (Open-Vocabulary Detector + SAM) within the rectified patch to extract precise, instance-level masks.

<div align="center">
    <img src="assets/pipeline.svg" alt="pipeline" width="90%">
</div>

---

## 📊 Dataset (PAP-12K)

PAP-12K is explicitly designed to encapsulate the unique challenges of 360° Equirectangular Projection (ERP) imagery:
- **Geometric Distortion:** Objects suffer from severe stretching near the poles.
- **Extreme Scale Variations:** Unconstrained environments lead to minute, sub-scale interactive targets.
- **Boundary Discontinuity:** Continuous objects are split at image edges.

*(Dataset download links and formatting instructions will be provided here soon.)*

---

## 🚀 Getting Started

**The codebase is currently undergoing internal review and clean-up.**

We plan to release the following components soon:
- [ ] PAP-12K Dataset (Images, QA annotations, and Segmentation Masks)
- [ ] Evaluation Scripts for the benchmark
- [ ] Source Code for the PAP inference pipeline

Please stay tuned for updates!

---

## 📝 Citation
If you find our dataset or method helpful in your research, please consider citing:

```bibtex
@article{pap2026,
  title={Panoramic Affordance Prediction}, 
  author={Anonymous Authors},
  journal={Coming Soon},
  year={2026}
}
```