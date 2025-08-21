from traffic_encoder_3d import DFEncoder
from logic_encoder_8d import LogicFeatureEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.6         # reject threshold
N_CLOSED  = 1600
N_OPEN    = 1600

# ========================================
traffic_encoder = DFEncoder(input_length=5000).to(device)
logic_encoder = LogicFeatureEncoder().to(device)
projection_traffic = nn.Linear(512, 256).to(device)
projection_logic = nn.Linear(512, 256).to(device)

ckpt = torch.load('STAR_model_pt/best_STAR_model.pt')
traffic_encoder.load_state_dict(remove_module_prefix(ckpt['traffic_encoder']))
logic_encoder.load_state_dict(remove_module_prefix(ckpt['logic_encoder']))
projection_traffic.load_state_dict(remove_module_prefix(ckpt['projection_traffic']))
projection_logic.load_state_dict(remove_module_prefix(ckpt['projection_logic']))

traffic_encoder.eval()
logic_encoder.eval()
projection_traffic.eval()
projection_logic.eval()

# ==================== monitored ====================
CACHE_CW = 'STAR_dataset/cached_dataset_cw_3_10.npz'
cache_cw = np.load(CACHE_CW)
logic_semantics = torch.tensor(cache_cw['logic'], dtype=torch.float32)
traffic_vectors = torch.tensor(cache_cw['traffic'], dtype=torch.float32)
label_cw = torch.tensor(cache_cw['label'], dtype=torch.long)

torch.manual_seed(42)
all_classes = torch.unique(label_cw)
sel_classes = all_classes[torch.randperm(len(all_classes))[:N_CLOSED]]
sel_idx = []
for cls in sel_classes.cpu().numpy():
    idx_pool = np.where(label_cw.cpu().numpy() == cls)[0]
    sel_idx.append(np.random.choice(idx_pool, 1)[0])
sel_idx = torch.tensor(sel_idx, dtype=torch.long)

logic_cw = logic_semantics[sel_idx]        # [1600, 80, 8]
traffic_cw = traffic_vectors[sel_idx]        # [1600, 1, 5000]
label_cw = label_cw[sel_idx]               # [1600]

# ==================== unmonitored ====================
CACHE_OW = 'STAR_dataset/cached_dataset_ow_extended.npz'
cache_ow = np.load(CACHE_OW)
traffic_ow_full = torch.tensor(cache_ow['traffic'], dtype=torch.float32)

perm = torch.randperm(traffic_ow_full.size(0))
traffic_ow = traffic_ow_full[perm[:N_OPEN]]

traffic_all = torch.cat([traffic_cw, traffic_ow], dim=0)          # [3200, 1, 5000]
is_closed = torch.zeros(traffic_all.size(0), dtype=torch.bool)
is_closed[:N_CLOSED] = True

# ========================================
with torch.no_grad():
    logic_emb = logic_encoder(logic_cw.to(device))
    logic_emb = projection_logic(logic_emb)
    logic_emb = F.normalize(logic_emb, dim=1)                      # [1600, 256]

    traffic_emb = traffic_encoder(traffic_all.to(device))
    traffic_emb = projection_traffic(traffic_emb)
    traffic_emb = F.normalize(traffic_emb, dim=1)                  # [3200, 256]

    sim_matrix = traffic_emb @ logic_emb.T                         # [3200, 1600]
    max_sim, pred_idx = sim_matrix.max(dim=1)

# ==================== binary cls ====================
pred_known = max_sim >= THRESHOLD

TP = ((pred_known &  is_closed.to(device))).sum().item()
FN = ((~pred_known &  is_closed.to(device))).sum().item()
FP = ((pred_known & ~is_closed.to(device))).sum().item()
TN = ((~pred_known & ~is_closed.to(device))).sum().item()

precision = TP / (TP + FP) if (TP + FP) else 0.0
recall = TP / (TP + FN) if (TP + FN) else 0.0

closed_mask = torch.arange(N_CLOSED, device=device)
correct_top1 = (pred_idx[:N_CLOSED] == closed_mask) & pred_known[:N_CLOSED]
top1_acc_known = correct_top1.float().mean().item() if TP else 0.0

top5_hit = (sim_matrix[:N_CLOSED].topk(5, dim=1).indices == closed_mask.unsqueeze(1)).any(dim=1)
correct_top5 = top5_hit & pred_known[:N_CLOSED]
top5_acc_known = correct_top5.float().mean().item() if TP else 0.0

# ========================================
print(f"[Closed‑World] sample num: {N_CLOSED}")
print(f"[Open‑World]   sample num: {N_OPEN}")
print(f"[Thresh]       {THRESHOLD:.2f}\n")

print("binary cls:")
print(f"  TP={TP:4d}  FN={FN:4d}  FP={FP:4d}  TN={TN:4d}")
print(f"  Precision = {precision:.4f}")
print(f"  Recall    = {recall:.4f}\n")

print("monitored 1600 cls:")
print(f"  Top‑1 Acc = {top1_acc_known:.4f}")
print(f"  Top‑5 Acc = {top5_acc_known:.4f}")