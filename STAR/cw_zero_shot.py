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

# load test data
CACHE_PATH = 'STAR_dataset/cached_dataset_cw_3_10.npz'

if os.path.exists(CACHE_PATH):
    print(f"[INFO] Loading cached dataset from {CACHE_PATH}")
    cache = np.load(CACHE_PATH)

    logic_semantics = torch.tensor(cache['logic'], dtype=torch.float32)
    traffic_vectors = torch.tensor(cache['traffic'], dtype=torch.float32)
    label = torch.tensor(cache['label'], dtype=torch.long)

    print(f'initial dataset shape: {logic_semantics.shape}, {traffic_vectors.shape}, {label.shape}')
else:
    raise ValueError(f"[ERROR] Dataset not found at {CACHE_PATH}")

# -------------------------------
# random select 1600 website
# -------------------------------
all_classes = torch.unique(label)

if len(all_classes) < 1600:
    raise ValueError(f"[ERROR] {len(all_classes)}, less than 1600")

# torch.manual_seed(42)

selected_classes = all_classes[torch.randperm(len(all_classes))[:1600]]

selected_idx_logic = []
selected_idx_traffic = []

labels_np = label.cpu().numpy()

for cls in selected_classes.cpu().numpy():
    idx = np.where(labels_np == cls)[0]

    idx_sel_logic = np.random.choice(idx, 1)[0]
    selected_idx_logic.append(idx_sel_logic)

    idx_sel_traffic = np.random.choice(idx, 1)[0]
    selected_idx_traffic.append(idx_sel_traffic)


selected_idx_logic = torch.tensor(selected_idx_logic, dtype=torch.long)
selected_idx_traffic = torch.tensor(selected_idx_traffic, dtype=torch.long)

new_logic_semantics = logic_semantics[selected_idx_logic]
new_traffic_vectors = traffic_vectors[selected_idx_traffic]
new_label = label[selected_idx_traffic]

print(f"used dataset shape: {new_logic_semantics.shape}, {new_traffic_vectors.shape}, {new_label.shape}")
print(f"used label size: {len(torch.unique(new_label))}")


logic_encoder.eval()
traffic_encoder.eval()
projection_traffic.eval()
projection_logic.eval()

with torch.no_grad():
    traffic_input = new_traffic_vectors.to(device)  # [B, 1, 1500]
    logic_input = new_logic_semantics.to(device)

    traffic_embeds = traffic_encoder(traffic_input)  # [B, 512]
    traffic_embeds = projection_traffic(traffic_embeds)  # [B, 128]
    logic_embeds = logic_encoder(logic_input)  # [B, 128]
    logic_embeds = projection_logic(logic_embeds)

    logic_embed_norm = F.normalize(logic_embeds, dim=1)
    traffic_embed_norm = F.normalize(traffic_embeds, dim=1)

    similarity = traffic_embed_norm @ logic_embed_norm.T  # [N, N]
    top1 = (similarity.argmax(dim=1) == torch.arange(similarity.size(0), device=device)).float().mean()
    top5 = (similarity.topk(5, dim=1).indices == torch.arange(similarity.size(0), device=device).unsqueeze(1)).any(
        dim=1).float().mean()

    print(f"[VAL] Top-1 acc: {top1.item():.4f}, Top-5 acc: {top5.item():.4f}")


