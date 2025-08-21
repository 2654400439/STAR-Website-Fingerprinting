from traffic_encoder_3d import DFEncoder
from logic_encoder_8d import LogicFeatureEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np

def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def safe_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def batched_encode(encoder, projection, input_tensor, batch_size=128):
    encoder.eval()
    projection.eval()

    outputs = []
    with torch.no_grad():
        for i in range(0, input_tensor.size(0), batch_size):
            batch = input_tensor[i:i + batch_size]
            feats = encoder(batch)
            feats = projection(feats)
            feats = F.normalize(feats, dim=-1)
            outputs.append(feats)

    return torch.cat(outputs, dim=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

traffic_encoder = DFEncoder(input_length=5000).to(device)
logic_encoder = LogicFeatureEncoder().to(device)
projection_traffic = nn.Linear(512, 256).to(device)
projection_logic = nn.Linear(512, 256).to(device)

checkpoint = torch.load('STAR_model_pt/best_STAR_model.pt')
logic_encoder.load_state_dict(remove_module_prefix(checkpoint['logic_encoder']))
traffic_encoder.load_state_dict(remove_module_prefix(checkpoint['traffic_encoder']))
projection_logic.load_state_dict(remove_module_prefix(checkpoint['projection_logic']))
projection_traffic.load_state_dict(remove_module_prefix(checkpoint['projection_traffic']))

# ========================================
CACHE_PATH = 'STAR_dataset/cached_dataset_cw_1600_40.npz'

if os.path.exists(CACHE_PATH):
    print(f"[INFO] Loading cached dataset from {CACHE_PATH}")
    cache = np.load(CACHE_PATH)
    print(f'dataset shape: {cache["logic"].shape}, {cache["traffic"].shape}, {cache["label"].shape}')
else:
    raise ValueError

# ========================================
torch.manual_seed(42)

labels_np_full = cache['label']
all_classes = np.unique(labels_np_full)

if len(all_classes) < 1600:
    raise ValueError

selected_classes = np.random.choice(all_classes, 1600, replace=False)

mask = np.isin(labels_np_full, selected_classes)

logic_semantics = cache['logic'][mask]
traffic_vectors = cache['traffic'][mask]
labels_np = labels_np_full[mask]

print(f"filtered shape: {logic_semantics.shape}, {traffic_vectors.shape}, {labels_np.shape}")

# ========================================
num_support_per_class = 8
alpha = 5  # tip-adapter param

unique_classes = np.unique(labels_np)

support_idx = []
query_idx = []

for cls in unique_classes:
    idx_cls = np.where(labels_np == cls)[0]
    if len(idx_cls) < num_support_per_class + 1:
        continue
    np.random.shuffle(idx_cls)
    support_idx.extend(idx_cls[:num_support_per_class])
    query_idx.extend(idx_cls[num_support_per_class:])

support_logic = torch.tensor(logic_semantics[support_idx]).float().to(device)
support_traffic = torch.tensor(traffic_vectors[support_idx]).float().to(device)
support_labels = torch.tensor(labels_np[support_idx]).long().to(device)

query_logic = torch.tensor(logic_semantics[query_idx]).float().to(device)
query_traffic = torch.tensor(traffic_vectors[query_idx]).float().to(device)
query_labels = torch.tensor(labels_np[query_idx]).long().to(device)

print(f"support: {support_traffic.shape}, query: {query_traffic.shape}")

# Prototype
prototype_idx = []
for cls in unique_classes:
    idx_cls = np.where(labels_np == cls)[0]
    if len(idx_cls) == 0:
        continue
    prototype_idx.append(np.random.choice(idx_cls, 1)[0])

prototype_logic = torch.tensor(logic_semantics[prototype_idx]).float().to(device)
prototype_labels = torch.tensor(labels_np[prototype_idx]).long().to(device)

print(f"Prototype shape: {prototype_logic.shape}")

# ========================================
with torch.no_grad():
    # Cache keys
    cache_embeds = batched_encode(traffic_encoder, projection_traffic, support_traffic)

    # Prototype
    proto_embeds = batched_encode(logic_encoder, projection_logic, prototype_logic)

    # Query
    query_embeds = batched_encode(traffic_encoder, projection_traffic, query_traffic)

    # Zero-shot logits
    logits_proto = query_embeds @ proto_embeds.T  # [Q, P]

    # Cache keys logits
    sim_cache = query_embeds @ cache_embeds.T  # [Q, S]

    logits_cache = torch.zeros(query_embeds.size(0), len(unique_classes)).to(device)

    class_map = {old: new for new, old in enumerate(unique_classes)}
    mapped_support_labels = torch.tensor([class_map[l.item()] for l in support_labels]).to(device)
    mapped_prototype_labels = torch.tensor([class_map[l.item()] for l in prototype_labels]).to(device)
    mapped_query_labels = torch.tensor([class_map[l.item()] for l in query_labels]).to(device)

    for i, cls in enumerate(unique_classes):
        mask = (mapped_support_labels == i).float()
        logits_cache[:, i] = (sim_cache * mask).sum(dim=1) / (mask.sum() + 1e-6)

    logits = logits_proto + alpha * logits_cache
    preds = logits.argmax(dim=1)

    acc = (preds == mapped_query_labels).float().mean()

    top5_preds = logits.topk(5, dim=1).indices
    matches = (top5_preds == mapped_query_labels.unsqueeze(1))
    top5_acc = matches.any(dim=1).float().mean()

print(f"Tip-Adapter Few-Shot Acc: {acc.item():.4f}， Top-5: {top5_acc.item():.4f}")
