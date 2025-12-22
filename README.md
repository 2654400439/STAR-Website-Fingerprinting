# STAR-Website-Fingerprinting

[English](README.md) | [中文](README_zh.md)


![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
[![conference](https://img.shields.io/badge/INFOCOM-2026-orange)](https://infocom2026.ieee-infocom.org/)
[![task](https://img.shields.io/badge/task-Zero--shot%20WF-purple)](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=website+fingerprinting&oq=website+)




<p align="center">
  <img src="./images/logo.png" width="400">
</p>

The code and dataset for the paper **STAR: Semantic-Traffic Alignment and Retrieval for Zero-Shot HTTPS Website Fingerprinting**, 
accepted in *IEEE International Conference on Computer Communications (INFOCOM) 2026*.  

- 📄 [Read the Camera-Ready Paper](docs/STAR_infocom26_1137_rfp.pdf)  
- 🌐 [Read on arXiv](https://arxiv.org/abs/2512.17667)

⚠️ **For research purposes only.** ⚠️

If you find this repository useful, please cite our paper:

```bibtex
@article{cheng2025star,
  title={STAR: Semantic-Traffic Alignment and Retrieval for Zero-Shot HTTPS Website Fingerprinting},
  author={Yifei Cheng and Yujia Zhu and Baiyang Li and Xinhao Deng and Yitong Cai and Yaochen Ren and Qingyun Liu},
  journal={arXiv preprint arXiv:2512.17667},
  year={2025}
}
```

*The official IEEE INFOCOM version will be updated once published.*


The processed dataset and pretrained checkpoints are publicly available via [Zenodo](https://doi.org/10.5281/zenodo.17060855)

---

## Reproducibility

This section provides step-by-step instructions to reproduce the main experimental results reported in the paper.

### 1. Environment Setup

All experiments are implemented in Python.  
Please first install the required dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

> We recommend using a dedicated virtual environment (e.g., `venv` or `conda`) to avoid dependency conflicts.

### 2. Dataset and Pretrained Model

We provide the **processed dataset** and **pretrained model checkpoints** required for reproduction via a publicly accessible Zenodo repository.

#### Required Files and Directory Structure

Please organize the downloaded files as follows:

```text
STAR/
├── STAR_dataset/
│   ├── (processed dataset files)
│   └── .gitkeep
├── STAR_model_pt/
│   ├── best_STAR_model.pt
│   └── .gitkeep
```

#### Pretrained Model

- Download `best_STAR_model.pt`
- Place it at:
  ```text
  /STAR_model_pt/best_STAR_model.pt
  ```
  
> 🔗 **Zenodo link:** https://doi.org/10.5281/zenodo.17060855

#### Notes on Data Availability

The dataset released in this repository is **preprocessed according to the input format required by STAR**, as described in the paper.

The **raw data** used in this work—including:

- over **170,000 website visits**,
- more than **100 GB** of raw traffic traces (PCAP format),
- and corresponding logic-side crawl logs—

is not publicly hosted due to storage and distribution constraints.
If access to the raw data is required for research purposes, please contact:

> 📧 chengyifei@iie.ac.cn


### 3. Running Experiments

All experiment scripts are located in the project root directory:

```text
STAR/
├── cw_zero_shot.py
├── cw_linear_probe.py
├── cw_tip_adapter.py
├── ow_zero_shot.py
├── pretrain.py
├── logic_encoder_8d.py
├── traffic_encoder_3d.py
```

We categorize experiments by **filename prefixes**.

#### 3.1 Closed-World Experiments (`cw_*.py`)

Scripts with the prefix `cw_` correspond to **closed-world evaluation**, including:

- **Zero-shot classification**

  ```bash
  python cw_zero_shot.py
  ```


- **Few-shot adaptation**

  - Linear probing

    ```bash
    python cw_linear_probe.py
    ```

  - Tip-Adapter-style adaptation

    ```bash
    python cw_tip_adapter.py
    ```


These scripts reproduce the closed-world results reported in the paper.

#### 3.2 Open-World Experiments (`ow_*.py`)

Scripts with the prefix `ow_` correspond to **open-world evaluation**, including rejection of unmonitored websites.

  ```bash
  python ow_zero_shot.py
  ```

### 4. Model Pretraining (Optional)

Users may also choose to **pretrain the STAR model from scratch** using the provided training script:

  ```bash
  python pretrain.py
  ```

#### Training Configuration

- Training follows the data scale and optimization strategy described in the paper.

- Default setting:

  - **200 epochs**

  - Approximately **4 hours** using **5 NVIDIA A100 GPUs** with data parallelism.

> ⚠️ Pretraining is computationally expensive and **not required** for reproducing the main results, as pretrained checkpoints are provided.


### 5. Additional Notes

- All random seeds are fixed by default for reproducibility.

- GPU acceleration is recommended for both pretraining and evaluation.

If you encounter any issues during reproduction, feel free to open an issue or contact the authors.