<h1 align="center"> Panoramic Affordance Prediction </h1>

<div align="center">

[Zixin Zhang](https://scholar.google.com/citations?user=BbZ0mwoAAAAJ&hl=zh-CN)<sup>1*</sup>, [Chenfei Liao](https://chenfei-liao.github.io/)<sup>1*</sup>, [Hongfei Zhang](https://github.com/soyouthinkyoucantell)<sup>1</sup>, [Harold H. Chen](https://haroldchen19.github.io/)<sup>1</sup>, [Kanghao Chen](https://scholar.google.com/citations?hl=zh-CN&user=IwvcylUAAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup>, [Zichen Wen](https://scholar.google.com/citations?user=N-aPFvEAAAAJ&hl=zh-CN&oi=ao)<sup>3</sup>, [Litao Guo](https://scholar.google.com/citations?hl=zh-CN&user=efdm760AAAAJ)<sup>1</sup>, [Bin Ren](https://amazingren.github.io/)<sup>4</sup>, [Xu Zheng](https://zhengxujosh.github.io/)<sup>1</sup>, [Yinchuan Li](https://yinchuanll.github.io/)<sup>6</sup>, [Xuming Hu](https://xuminghu.github.io/)<sup>1</sup>, [Nicu Sebe](https://disi.unitn.it/~sebe/)<sup>5</sup>, [Ying-Cong Chen](https://www.yingcong.me/)<sup>1,2&dagger;</sup>

<sup>1</sup>HKUST(GZ), <sup>2</sup>HKUST, <sup>3</sup>SJTU, <sup>4</sup>MBZUAI, <sup>5</sup>University of Trento, <sup>6</sup>Knowin

<small>*Equal contribution &nbsp;&nbsp;&nbsp; &dagger;Corresponding author</small>

</div>

<div align="center">
    <a href="https://zixinzhang02.github.io/Panoramic-Affordance-Prediction/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github&logoColor=white" alt="Project Page"></a>
    <a href="#"><img src="https://img.shields.io/badge/Paper-Coming_Soon-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
    <a href="#"><img src="https://img.shields.io/badge/Dataset-Coming_Soon-orange?style=for-the-badge&logo=huggingface&logoColor=white" alt="Dataset"></a>
</div>
<br>

Official repository for the paper: **Panoramic Affordance Prediction**.

> Affordance prediction serves as a critical bridge between perception and action in embodied AI. However, existing research is confined to pinhole camera models, which suffer from narrow Fields of View (FoV) and fragmented observations. In this paper, we present the first exploration into **Panoramic Affordance Prediction**, utilizing 360-degree imagery to capture global spatial relationships and holistic scene understanding. 



<br>

<div align="center">
    <img src="assets/teaser.svg" alt="teaser" width="90%">
</div>


## 🚀 Right Around the Corner!

**The codebase is currently undergoing internal review and clean-up.**

We plan to release the following components soon (**in two weeks**):
- [ ] PAP-12K Dataset (All full resolution images, QA annotations, and Segmentation Masks)
- [ ] Evaluation Scripts for the benchmark
- [ ] Source Code for the PAP inference pipeline

***Please stay tuned for updates!***

---

## 🌟 Highlights
- **New Task:** We introduce the **First Exploration** into **Panoramic Affordance Prediction**, overcoming the "tunnel vision" of traditional pinhole camera based affordance methods.
- **PAP-12K Dataset (100% Real-World):** A large-scale benchmark featuring 1,003 natively captured ultra-high-resolution (12K) panoramic images from diverse indoor environments, coupled with over 13,000 carefully annotated reasoning-based QA pairs with pixel-level affordance masks.
- **PAP Framework:** A training-free, coarse-to-fine pipeline mimicking human foveal vision to handle panoramic challenges like geometric distortion, scale variations, and boundary discontinuity.

---

## 📊 Dataset (PAP-12K)

PAP-12K is explicitly designed to encapsulate the unique challenges of 360° Equirectangular Projection (ERP) imagery. Unlike synthetic or web-crawled datasets, **all 1,003 ultra-high resolution (11904×5952) panoramic images in PAP-12K were natively captured in real-world environments using professional 360° cameras.** This ensures authentic geometric distortions, lighting conditions, and natural object scales, bridging the gap between static dataset evaluation and practical robotic applications.

Key challenges captured include:
- **Geometric Distortion:** Objects suffer from severe stretching near the poles.
- **Extreme Scale Variations:** Unconstrained environments lead to minute, sub-scale interactive targets.
- **Boundary Discontinuity:** Continuous objects are split at image edges.

<div align="center">
    <img src="assets/challenge.png" alt="challenge" width="90%">
</div>

*(Dataset download links and formatting instructions will be provided here soon.)*

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

## 📧 Contact
If you have any questions or suggestions, please feel free to contact us at [zzhang300@connect.hkust-gz.edu.cn](mailto:zzhang300@connect.hkust-gz.edu.cn), [cliao127@connect.hkust-gz.edu.cn](mailto:cliao127@connect.hkust-gz.edu.cn).