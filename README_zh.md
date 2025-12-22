# STAR-Website-Fingerprinting

[English](README.md) | [中文](README_zh.md)

![python](https://img.shields.io/badge/python-3.9%2B-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
[![conference](https://img.shields.io/badge/INFOCOM-2026-orange)](https://infocom2026.ieee-infocom.org/)
[![task](https://img.shields.io/badge/task-Zero--shot%20WF-purple)](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=website+fingerprinting&oq=website+)

<p align="center">
  <img src="./images/logo.png" width="400">
</p>

本仓库提供论文 **STAR: Semantic-Traffic Alignment and Retrieval for Zero-Shot HTTPS Website Fingerprinting** 的代码与数据集，
该论文已被 *IEEE International Conference on Computer Communications (INFOCOM) 2026* 接收。

- 📄 [阅读 Camera-Ready 版本](docs/STAR_infocom26_1137_rfp.pdf)  
- 🌐 [在 arXiv 上阅读](https://arxiv.org/abs/2512.17667)

⚠️ **仅供科研用途。** ⚠️

如果你觉得本仓库对你有帮助，请引用我们的论文：

```bibtex
@article{cheng2025star,
  title={STAR: Semantic-Traffic Alignment and Retrieval for Zero-Shot HTTPS Website Fingerprinting},
  author={Yifei Cheng and Yujia Zhu and Baiyang Li and Xinhao Deng and Yitong Cai and Yaochen Ren and Qingyun Liu},
  journal={arXiv preprint arXiv:2512.17667},
  year={2025}
}
```

*IEEE INFOCOM 的正式版本在公开出版后会在此更新。*

处理后的数据集与预训练模型 checkpoint 已通过 [Zenodo](https://doi.org/10.5281/zenodo.17060855) 公开发布。

---

## 可复现性说明

本节提供复现实验论文中主要结果的逐步操作说明。

### 1. 环境配置

所有实验均基于 Python 实现。
请先安装 `requirements.txt` 中列出的依赖：

```bash
pip install -r requirements.txt
```

> 建议使用独立虚拟环境（如 `venv` 或 `conda`）以避免依赖冲突。

### 2. 数据集与预训练模型

我们通过公开的 Zenodo 仓库提供复现实验所需的**处理后数据集**与**预训练模型 checkpoint**。

#### 必需文件与目录结构

请将下载的文件按如下结构组织：

```text
STAR/
├── STAR_dataset/
│   ├── (processed dataset files)
│   └── .gitkeep
├── STAR_model_pt/
│   ├── best_STAR_model.pt
│   └── .gitkeep
```

### 预训练模型

- 下载 `best_STAR_model.pt`

- 放置路径为：

```text
/STAR_model_pt/best_STAR_model.pt
```

> 🔗 **Zenodo 链接**： https://doi.org/10.5281/zenodo.17060855


#### 数据可用性说明

本仓库发布的数据集已**按照 STAR 所需输入格式完成预处理**（详见论文描述）。

本工作使用的**原始数据**包括：

- 超过 **170,000 次网站访问**，

- 超过 **100 GB** 的原始流量（PCAP 格式），

- 以及对应的逻辑侧爬取日志（crawl logs），

由于存储与分发成本限制，暂不在公开平台托管。如科研需要获取原始数据，请联系：

> 📧 chengyifei@iie.ac.cn


### 3. 运行实验

所有实验脚本均位于项目根目录：

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


我们按**文件名前缀**对实验脚本进行分类。

#### 3.1 封闭世界实验（`cw_*.py`）

以 `cw_` 开头的脚本对应**封闭世界评估**，包括：

- **零样本分类**

```bash
python cw_zero_shot.py
```

- **小样本适配（few-shot adaptation）**

    - 线性探测（linear probing）

    ```bash
  python cw_linear_probe.py
    ```

    - Tip-Adapter 风格适配

    ```bash
  python cw_tip_adapter.py
    ```

上述脚本可复现论文中报告的封闭世界实验结果。

#### 3.2 开放世界实验（`ow_*.py`）

以 `ow_` 开头的脚本对应开放世界评估，包含对未监控网站的拒识（rejection）。

```bash
python ow_zero_shot.py
```

### 4. 模型预训练（可选）

你也可以选择使用提供的训练脚本**从零开始预训练** STAR 模型：

```bash
python pretrain.py
```


#### 训练配置说明

- 训练数据规模与优化策略与论文描述一致。

- 默认设置为：

    - **200 epochs**

    - 使用数据并行在 **5 张 NVIDIA A100 GPU** 上训练约 **4 小时**。

> ⚠️ 预训练计算开销较大，但**复现论文主要结果不需要从头预训练**（我们已提供预训练 checkpoint）。

### 5. 其他说明

- 默认固定所有随机种子，确保可复现性。

- 推荐使用 GPU 加速（无论预训练还是评估）。

如在复现过程中遇到问题，欢迎提 issue 或联系作者。