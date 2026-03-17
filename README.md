<h1 align="center"> Panoramic Affordance Prediction </h1>

<div align="center">

[Zixin Zhang](https://scholar.google.com/citations?user=BbZ0mwoAAAAJ&hl=zh-CN)<sup>1*</sup>, [Chenfei Liao](https://chenfei-liao.github.io/)<sup>1*</sup>, [Hongfei Zhang](https://github.com/soyouthinkyoucantell)<sup>1</sup>, [Harold H. Chen](https://haroldchen19.github.io/)<sup>1</sup>, [Kanghao Chen](https://scholar.google.com/citations?hl=zh-CN&user=IwvcylUAAAAJ&view_op=list_works&sortby=pubdate)<sup>1</sup>, [Zichen Wen](https://scholar.google.com/citations?user=N-aPFvEAAAAJ&hl=zh-CN&oi=ao)<sup>3</sup>, [Litao Guo](https://scholar.google.com/citations?hl=zh-CN&user=efdm760AAAAJ)<sup>1</sup>, [Bin Ren](https://amazingren.github.io/)<sup>4</sup>, [Xu Zheng](https://zhengxujosh.github.io/)<sup>1</sup>, [Yinchuan Li](https://yinchuanll.github.io/)<sup>6</sup>, [Xuming Hu](https://xuminghu.github.io/)<sup>1</sup>, [Nicu Sebe](https://disi.unitn.it/~sebe/)<sup>5</sup>, [Ying-Cong Chen](https://www.yingcong.me/)<sup>1,2&dagger;</sup>

<sup>1</sup>HKUST(GZ), <sup>2</sup>HKUST, <sup>3</sup>SJTU, <sup>4</sup>MBZUAI, <sup>5</sup>UniTrento, <sup>6</sup>Knowin

<small>*Equal contribution &nbsp;&nbsp;&nbsp; &dagger;Corresponding author</small>


</div>

<div align="center">
    <a href="https://zixinzhang02.github.io/Panoramic-Affordance-Prediction/"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github&logoColor=white" alt="Project Page"></a>
    <a href="#"><img src="https://img.shields.io/badge/Dataset-Download_ZIP-orange?style=for-the-badge&logo=icloud&logoColor=white" alt="Dataset"></a>
    <a href="https://zixinzhang02.github.io/Panoramic-Affordance-Prediction/static/papers/Paper_high_res.pdf"><img src="https://img.shields.io/badge/Paper_(High--res)-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
    <a href="https://arxiv.org/abs/2603.15558"><img src="https://img.shields.io/badge/Paper_(arXiv)-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
</div>
<br>

Official repository for the paper: **Panoramic Affordance Prediction**.

> Affordance prediction serves as a critical bridge between perception and action in the embodied AI era. However, existing research is confined to pinhole camera models, which suffer from narrow Fields of View (FoV) and fragmented observations. In this paper, we present the first exploration into **Panoramic Affordance Prediction**, utilizing 360-degree imagery to capture global spatial relationships and holistic scene understanding. 



<br>

<div align="center">
    <img src="assets/teaser.svg" alt="teaser" width="90%">
</div>


## 🚀 News
* **[2026-03-16]** 🔥 [PAP-12K Dataset](https://zixinzhang02.github.io/Panoramic-Affordance-Prediction/#dataset-preview) and [PAP Inference Code](https://github.com/EnVision-Research/PAP) are released! Welcome to try it out!
* **[2026-03-14]** 📄 [Paper](https://zixinzhang02.github.io/Panoramic-Affordance-Prediction/static/papers/Paper_high_res.pdf) is released.
* **[2026-03-11]** 🌐 [Repository](https://github.com/EnVision-Research/PAP) and [Webpage](https://zixinzhang02.github.io/Panoramic-Affordance-Prediction/) are released.

---

## 🌟 Highlights
- **New Task:** We introduce the **First Exploration** into **Panoramic Affordance Prediction**, overcoming the "tunnel vision" of traditional pinhole camera based affordance methods.
- **PAP-12K Dataset (100% Real-World):** A large-scale benchmark featuring 1,003 natively captured ultra-high-resolution (12K) panoramic images from diverse indoor environments, coupled with over 13,000 carefully annotated reasoning-based QA pairs with pixel-level affordance masks.
- **PAP Framework:** A training-free, coarse-to-fine pipeline mimicking human foveal vision to handle panoramic challenges like geometric distortion, scale variations, and boundary discontinuity.

---

## 🛠️ Environment Setup
### 1. Download the models

```
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct
huggingface-cli download IDEA-Research/Rex-Omni
huggingface-cli download facebook/sam2.1-hiera-large
```
### 2. Install Dependencies
```
conda create -n pap python=3.11
conda activate pap
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
Flash Attention is required for Rex-Omni. We strongly recommend installing Flash Attention using a pre-built wheel to avoid compilation issues.

You can find the pre-built wheel for your system [here](https://github.com/Dao-AILab/flash-attention/releases). For the environment setup above, use:
```
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```
Then, install the dependencies.
```
pip install -r requirements.txt
pip install git+https://github.com/IDEA-Research/Rex-Omni.git --no-deps
pip install git+https://github.com/facebookresearch/sam2.git
```
---


## 🚀 Quick Demo
First, use vllm to deploy the model. Qwen3-VL-32B model requires about 60~70 GB GPU memory when deployed with vllm, you can adjust the `tensor-parallel-size` according to your GPU memory.

> PAP is a highly adaptive framework. We use Qwen3-VL-32B as our validated default VLM, but you can quickly swap it for any other local VLM. As long as a model is compatible with vLLM and meets a basic quality threshold, it can be integrated into this pipeline directly with solid results.
```
vllm serve Qwen/Qwen3-VL-32B-Instruct --served-model-name qwen3-vl-32b --port 8088 --max_model_len 20000 --tensor-parallel-size 1
```
Then, run the demo code to inference on the provided image and question (or you can put your own image and question here).
```
cd demo
python demo.py \
    --vlm_api_url "http://localhost:8088" \
    --vlm_model_name "qwen3-vl-32b" \
    --image_path "kitchen.jpg" \
    --question_file "kitchen.txt" \
    --output "kitchen_output"
```
---

## 📊 PAP-12K Dataset
### Dataset Preview
We provide a preview of the PAP-12K Dataset in [Dataset-Preview](https://zixinzhang02.github.io/Panoramic-Affordance-Prediction/#dataset-preview). You can check the preview of the dataset before downloading. 

### Downloading with Cloud Drive
1. [Google Drive](https://drive.google.com/file/d/1Bq4wLL9AoSBP1Im545qKWlk85cP21VQE/view?usp=sharing)
2. [Baidu Netdisk](https://pan.baidu.com/s/1FeNdQ67vkfUYX0qXerInTw?pwd=u8vd)

### Dataset Structure
You can refer to utils/dataset_utils.py for reading the dataset. The dataset structure is as follows:
```
PAP-12K
├── balcony/
├──── 0001/
├────── washing_machine/
├──────── mask.png
├──────── affordance_question.txt
├────── faucet/
├────── ...
├────── 0001.jpg
├──── 0002/
├──── ...
├── bathroom/
├── bedroom/
├── ...
```
---

## 🚀 Inference on PAP-12K
```shell
vllm serve Qwen/Qwen3-VL-32B-Instruct --served-model-name qwen3-vl-32b --port 8088 --max_model_len 20000 --tensor-parallel-size 1
```
```
python run.py \
    --dataset_root /path/to/PAP-12K \
    --output output/PAP \
    --vlm_api_url http://localhost:8088 \
    --vlm_model_name qwen3-vl-32b \
    --vlm_concurrency 8 \
    --resume
```

---
## 💬 Citation
```
@article{zhang2026pap,
    title={Panoramic Affordance Prediction}, 
    author={Zhang, Zixin and Liao, Chenfei and Zhang, Hongfei and Chen, Harold Haodong and Chen, Kanghao and Wen, Zichen and Guo, Litao and Ren, Bin and Zheng, Xu and Li, Yinchuan and Hu, Xuming and Sebe, Nicu and Chen, Ying-Cong},
    journal={arXiv preprint arXiv:2603.15558},
    year={2026}
  }
```

---

## 📧 Contact
If you have any questions or suggestions, please feel free to contact us at [zzhang300@connect.hkust-gz.edu.cn](mailto:zzhang300@connect.hkust-gz.edu.cn), [cliao127@connect.hkust-gz.edu.cn](mailto:cliao127@connect.hkust-gz.edu.cn).
