from traffic_encoder_3d import DFEncoder
from logic_encoder_8d import LogicFeatureEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.preprocessing import LabelEncoder

import os
import numpy as np


def remove_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def safe_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def filter_and_sample(x, y, min_samples=20, target_classes=1600, seed=42):
    label_counts = Counter(y)

    valid_classes = [label for label, count in label_counts.items() if count >= min_samples]

    if len(valid_classes) < target_classes:
        raise ValueError

    np.random.seed(seed)
    selected_classes = np.random.choice(valid_classes, size=target_classes, replace=False)
    selected_classes_set = set(selected_classes)

    mask = np.isin(y, list(selected_classes_set))
    x_filtered = x[mask]
    y_filtered = y[mask]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_filtered)

    return x_filtered, y_encoded

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

CACHE_PATH = 'STAR_dataset/cached_dataset_cw_1600_40.npz'

if os.path.exists(CACHE_PATH):
    print(f"[INFO] Loading cached dataset from {CACHE_PATH}")
    cache = np.load(CACHE_PATH)

    traffic_vectors = torch.tensor(cache['traffic'], dtype=torch.float32)
    labels = torch.tensor(cache['label'], dtype=torch.long)
    print(f'dataset shape: {cache["logic"].shape}, {cache["traffic"].shape}, {cache["label"].shape}')
else:
    raise ValueError

torch.manual_seed(42)

all_classes = torch.unique(labels)

if len(all_classes) < 1600:
    raise ValueError

selected_classes = all_classes[torch.randperm(len(all_classes))[:1601]]

mask = torch.isin(labels, selected_classes)
traffic_vectors = traffic_vectors[mask]
labels = labels[mask]


# ==============================================================
num_classes = len(torch.unique(labels))
num_support = 8  # K-shot

labels_np = labels.numpy()
train_idx, val_idx = [], []

for cls in np.unique(labels_np):
    idx_cls = np.where(labels_np == cls)[0]
    if len(idx_cls) < num_support + 1:
        continue
    np.random.shuffle(idx_cls)
    train_idx.extend(idx_cls[:num_support])
    val_idx.extend(idx_cls[num_support:])

train_traffic = traffic_vectors[train_idx]
train_labels = labels[train_idx]

val_traffic = traffic_vectors[val_idx]
val_labels = labels[val_idx]

print(f"Train set: {train_traffic.shape}, Val set: {val_traffic.shape}")

# ========================================
unique_classes = torch.unique(train_labels)
class_map = {old.item(): new for new, old in enumerate(unique_classes)}

def remap(labels, class_map):
    labels_np = labels.numpy()
    new_labels_np = np.array([class_map[l] for l in labels_np])
    return torch.tensor(new_labels_np, dtype=torch.long)

train_labels = remap(train_labels, class_map)
val_labels = remap(val_labels, class_map)

print(f"[INFO] cls size: {torch.unique(train_labels)}")
print(f"[INFO] classifier output dim: {len(unique_classes)}")

projection_traffic.eval()
traffic_encoder.eval()


from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 128
EPOCHS = 100
feat_dim = 256

train_dataset = TensorDataset(train_traffic, train_labels)
val_dataset = TensorDataset(val_traffic, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

classifier = nn.Linear(feat_dim, num_classes).to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    classifier.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            embeds = traffic_encoder(batch_x)
            embeds = projection_traffic(embeds)

        logits = classifier(embeds)
        loss = F.cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f}")

    # ======= Validation =======
    classifier.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            embeds = traffic_encoder(batch_x)
            embeds = projection_traffic(embeds)
            logits = classifier(embeds)

            # Top-1
            preds = logits.argmax(dim=1)
            correct_top1 += (preds == batch_y).sum().item()

            # Top-5
            top5 = logits.topk(5, dim=1).indices
            match_top5 = (top5 == batch_y.unsqueeze(1)).any(dim=1)
            correct_top5 += match_top5.sum().item()

            total += batch_x.size(0)

    acc = correct_top1 / total
    top5_acc = correct_top5 / total
    print(f"Linear Probe | Top-1: {acc:.4f} | Top-5: {top5_acc:.4f}")
