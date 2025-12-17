# Technical Appendix: Inference and Adaptation Details

This document provides implementation-level details of the inference-stage procedures used in **STAR**, which are omitted from the main paper due to strict page limits. The appendix focuses on zero-shot cross-modal retrieval and two standard few-shot adaptation strategies.

Figures referenced in this appendix correspond to the inference-stage designs illustrated in the main paper (e.g., Fig. X(a–c)).

---

## A. Zero-Shot Inference via Cross-Modal Retrieval

### A.1 Overview

In the zero-shot setting, STAR identifies the website corresponding to an encrypted traffic trace without requiring any traffic samples from target websites during training. The task is formulated as a cross-modal retrieval problem between encrypted traffic traces and semantic logic profiles extracted via web crawling.

---

### A.2 Logic-Side Gallery Construction

For each monitored website class \( c \), a logic-side prototype embedding is pre-computed:

1. Extract the logic modality representation \( L_c \) from crawl-time browser logs.
2. Encode the logic representation using the trained logic encoder and projection head:
   \[
   \mathbf{z}^L_c = \frac{f_L(\mathrm{LogicEnc}(L_c))}{\|f_L(\mathrm{LogicEnc}(L_c))\|_2}
   \]
3. Store all \( \{\mathbf{z}^L_c\}_{c=1}^C \) as a gallery for retrieval.

This gallery is constructed offline and reused for all inference queries.

---

### A.3 Traffic Query Embedding

Given an encrypted traffic trace \( T \), the traffic-side embedding is computed as:
\[
\mathbf{z}^T = \frac{f_T(\mathrm{TrafficEnc}(T))}{\|f_T(\mathrm{TrafficEnc}(T))\|_2}
\]

The traffic encoder remains frozen during inference.

---

### A.4 Cross-Modal Retrieval and Classification

Cosine similarity is computed between the query embedding and each gallery entry:
\[
s_c = \cos(\mathbf{z}^T, \mathbf{z}^L_c) = \mathbf{z}^T \cdot \mathbf{z}^L_c
\]

The predicted class is obtained via nearest-neighbor retrieval:
\[
\hat{c} = \arg\max_c s_c
\]

---

### A.5 Open-World Rejection via Thresholding

To support open-world recognition, STAR applies a similarity threshold \( \tau \):

- If \( \max_c s_c \ge \tau \), the query is assigned to class \( \hat{c} \).
- Otherwise, the query is rejected as an unmonitored (unknown) website.

The threshold \( \tau \) is selected on a validation set to balance precision and recall.

---

## B. Few-Shot Adaptation via Linear Probe

### B.1 Setting

In the few-shot setting, a small labeled support set is provided:
\[
\mathcal{S} = \{(T_i, y_i)\}_{i=1}^{N}, \quad N = K \times C
\]
where \( K \) denotes the number of labeled samples per class.

The traffic encoder is frozen, and only a lightweight classifier is trained.

---

### B.2 Linear Classifier Training

Traffic embeddings are first extracted:
\[
\mathbf{z}^T_i = \mathrm{TrafficEnc}(T_i)
\]

A linear classifier \( g(\cdot) \) is trained on top of these embeddings using cross-entropy loss:
\[
\hat{y}_i = g(\mathbf{z}^T_i)
\]

Only the parameters of the linear layer are updated.

---

### B.3 Inference

For a test traffic trace \( T \), prediction is obtained as:
\[
\hat{y} = \arg\max g(\mathbf{z}^T)
\]

This strategy corresponds to standard linear probing commonly used in representation learning.

---

## C. Few-Shot Adaptation via Tip-Adapter

Tip-Adapter is a training-free adaptation strategy that combines zero-shot retrieval with few-shot memory-based inference.

---

### C.1 Memory Bank Construction

For each labeled support sample \( (T_i, y_i) \), its traffic embedding is computed:
\[
\mathbf{z}^T_i = \mathrm{TrafficEnc}(T_i)
\]

All embeddings are stored in a memory bank \( \mathcal{M} \).

---

### C.2 k-NN Logits from Few-Shot Samples

Given a test embedding \( \mathbf{z}^T \), similarity scores to the memory bank are computed:
\[
s_i = \cos(\mathbf{z}^T, \mathbf{z}^T_i)
\]

Class-wise k-NN logits are aggregated as:
\[
\ell^{\text{kNN}}_c = \sum_{i: y_i = c} \exp(\beta s_i)
\]
where \( \beta \) controls the sharpness of similarity weighting.

---

### C.3 Zero-Shot Anchor Logits

In parallel, STAR computes anchor logits from logic-side retrieval:
\[
\ell^{\text{ZS}}_c = \cos(\mathbf{z}^T, \mathbf{z}^L_c)
\]

---

### C.4 Logit Fusion

Final prediction logits are obtained via linear fusion:
\[
\ell_c = \ell^{\text{ZS}}_c + \alpha \cdot \ell^{\text{kNN}}_c
\]
where \( \alpha \) balances zero-shot alignment and few-shot evidence.

---

## D. Relationship Between Inference Paradigms

The three inference modes form a unified spectrum:

- **Zero-Shot Retrieval**: inference based solely on logic-side supervision.
- **Linear Probe**: supervised adaptation on frozen traffic representations.
- **Tip-Adapter**: hybrid inference combining retrieval-based alignment and few-shot memory.

These paradigms correspond to the three inference-stage designs illustrated in Fig. X(a–c) of the main paper.

---

## E. Practical Notes

- All similarity computations use cosine similarity on ℓ2-normalized embeddings.
- Logic-side gallery embeddings are pre-computed and cached offline.
- No retraining is required for zero-shot or Tip-Adapter inference.
- Inference complexity scales linearly with the number of monitored websites.
