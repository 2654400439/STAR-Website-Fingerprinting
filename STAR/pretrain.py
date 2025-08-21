import json.decoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import os
from tqdm import trange
import numpy as np
import sys

from feature_extraction.traffic_modal_process import process_traffic
from feature_extraction.logic_modal_process import process_logic

from traffic_encoder_3d import DFEncoder
from logic_encoder_8d import LogicFeatureEncoder

# ==================== training data prepare ====================
CACHE_PATH = 'STAR_dataset/cached_dataset_ow_log_traffic_logic_enhanced.npz'

def safe_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

if os.path.exists(CACHE_PATH):
    print(f"[INFO] Loading cached STAR_dataset from {CACHE_PATH}")
    cache = np.load(CACHE_PATH)
    cache_augment = np.load('STAR_dataset/cached_dataset_ow_augment.npz')
    all_logic = np.concatenate([cache['logic'], cache_augment['logic']], axis=0)
    all_traffic = np.concatenate([cache['traffic'], cache_augment['traffic']], axis=0)

    logic_semantics = torch.tensor(all_logic, dtype=torch.float32)
    traffic_vectors = torch.tensor(all_traffic, dtype=torch.float32)

    print(f'dataset scale: {all_logic.shape}, {all_traffic.shape}')
else:
    sys.exit(0)

# load supervised dataset
cache = np.load('STAR_dataset/cached_dataset_cw_1_2.npz')
traffic_cw = torch.tensor(cache['traffic'], dtype=torch.float32)
label_cw = torch.tensor(cache['label'], dtype=torch.long)
print(f'supervised dataset scale: {cache['traffic'].shape}, {cache['label'].shape}')


class WeakAlignDataset(torch.utils.data.Dataset):
    def __init__(self, logic_semantics, traffic_vectors):
        self.logic = torch.tensor(logic_semantics).float()  # [N, 80, 8]
        self.traffic = torch.tensor(traffic_vectors).float()  # [N, 3, 5000]

    def __len__(self):
        return len(self.logic)

    def __getitem__(self, idx):
        logic = self.logic[idx]   # [80, 8]
        traffic = self.traffic[idx]  # [3, 5000]
        label = torch.tensor(-1, dtype=torch.long)
        t_type = torch.tensor(0, dtype=torch.long)
        return logic, traffic, label, t_type


class LabeledTrafficDataset(torch.utils.data.Dataset):
    def __init__(self, traffic_vectors, labels):
        self.traffic = torch.tensor(traffic_vectors).float()  # [N, 3, 5000]
        self.labels = torch.tensor(labels).long().view(-1)    # [N]

    def __len__(self):
        return len(self.traffic)

    def __getitem__(self, idx):
        logic = torch.zeros(80, 8).float()
        traffic = self.traffic[idx]
        label = self.labels[idx]
        t_type = torch.tensor(0, dtype=torch.long)
        return logic, traffic, label, t_type


ds_weak = WeakAlignDataset(logic_semantics, traffic_vectors)
ds_label = LabeledTrafficDataset(traffic_cw, label_cw)
ds_mix = ConcatDataset([ds_weak, ds_label])

dataloader = DataLoader(ds_mix, batch_size=4096, shuffle=True, num_workers=4)


# ==================== validation data prepare ====================
file_folder = 'STAR_dataset/val_data/'
file_list = os.listdir(file_folder + '/browser_log')

logic_data_val, traffic_data_val = [], []
for j in trange(len(file_list)):
    try:
        semantic_info, flag = process_logic(f'{file_folder}/browser_log/{file_list[j]}')
    except json.decoder.JSONDecodeError:
        continue
    if flag == 0:
        continue
    try:
        traffic_info = process_traffic(f'{file_folder}/pcap/{file_list[j].split(".")[0]}.pcap', max_len=5000)
    except:
        continue

    logic_data_val.append(semantic_info)
    traffic_data_val.append(traffic_info)

logic_semantics_np_val = np.array(logic_data_val, dtype=np.float32)
traffic_vectors_np_val = np.array(traffic_data_val, dtype=np.float32)

print(f'validation dataset scale: {logic_semantics_np_val.shape}')

traffic_vectors_np_val[:, 0, :] = safe_log(traffic_vectors_np_val[:, 0, :])

logic_semantics_val = torch.tensor(logic_semantics_np_val, dtype=torch.float32)
traffic_vectors_val = torch.tensor(traffic_vectors_np_val, dtype=torch.float32)


# ==================== loss ====================
def clip_contrastive_loss(logic_embeds, traffic_embeds, temperature=0.07):
    if torch.isnan(logic_embeds).any():
        raise ValueError("Nan found in logic embeddings!")
    if torch.isnan(traffic_embeds).any():
        raise ValueError("Nan found in traffic embeddings!")

    logic_embeds = F.normalize(logic_embeds, dim=-1)
    traffic_embeds = F.normalize(traffic_embeds, dim=-1)

    # traffic as query
    logits = logic_embeds @ traffic_embeds.T  # [B, B]
    logits /= temperature


    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    final_loss = (loss_i2t + loss_t2i) / 2

    return final_loss


def traffic_intra_consistency_loss(embeds, labels):
    embeds = F.normalize(embeds, dim=1)
    sim_matrix = torch.matmul(embeds, embeds.T)
    mask = labels.view(-1,1) == labels.view(1,-1)
    mask = mask.float() - torch.eye(len(labels), device=labels.device)
    pos_pairs = sim_matrix * mask
    loss = (1 - pos_pairs).sum() / (mask.sum() + 1e-12)
    return loss


# ==================== pretraining ====================
def train_one_epoch(traffic_encoder, logic_encoder, projection_traffic, projection_logic, classifier_for_loss, dataloader, optimizer, device):
    traffic_encoder.train()
    logic_encoder.train()
    projection_traffic.train()
    projection_logic.train()
    classifier_for_loss.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_correct_webpage = 0
    total_samples_webpage = 0

    for logic_input, traffic_input, labels_, t_type in dataloader:
        logic_input = logic_input.to(device)  # [B, 80, 8]
        traffic_input = traffic_input.to(device)  # [B, 3, 5000]
        labels_ = labels_.to(device)  # [B]
        t_type = t_type.to(device) # [B]

        # Mask
        mask_infonce = labels_ == -1
        mask_website = t_type == 0
        mask_cls = labels_ != -1

        mask_website_infonce = mask_website & mask_infonce
        mask_website_cls = mask_website & mask_cls


        traffic_embeds = traffic_encoder(traffic_input)
        traffic_embeds = projection_traffic(traffic_embeds)
        logic_embeds = logic_encoder(logic_input)
        logic_embeds = projection_logic(logic_embeds)

        traffic_embeds_website_infonce = traffic_embeds[mask_website_infonce]
        logic_embeds_website_infonce = logic_embeds[mask_website_infonce]

        traffic_embeds_website_cls = traffic_embeds[mask_website_cls]
        labels_website_cls = labels_[mask_website_cls]

        # infonce loss
        loss_website_infonce = clip_contrastive_loss(
            logic_embeds_website_infonce, traffic_embeds_website_infonce
        ) if mask_website_infonce.any() else 0.0

        # consistency loss
        loss_website_cons = traffic_intra_consistency_loss(
            traffic_embeds_website_cls, labels_website_cls
        ) if mask_website_cls.any() else 0.0

        # cls loss
        logits_cls = classifier_for_loss(traffic_embeds_website_cls)
        loss_cls = F.cross_entropy(logits_cls, labels_website_cls) if mask_cls.any() else 0.0


        loss = loss_website_infonce + 0.02 * loss_website_cons + loss_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


        with torch.no_grad():
            # Normalize embeddings
            logic_norm = F.normalize(logic_embeds_website_infonce, dim=1)
            traffic_norm = F.normalize(traffic_embeds_website_infonce, dim=1)

            # Similarity matrix: [B, B]
            sim_matrix = torch.matmul(logic_norm, traffic_norm.T)

            # Prediction: find the index of max similarity
            preds = sim_matrix.argmax(dim=1)

            # Ground truth: assume i-th logic matches i-th traffic
            targets = torch.arange(sim_matrix.size(0)).to(device)

            correct = (preds == targets).sum().item()
            total = targets.size(0)

            total_correct += correct
            total_samples += total

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


# ==================== validation ====================
def validate_clip_model(logic_encoder, traffic_encoder, projection_traffic, projection_logic, logic_semantics, traffic_vectors, device):
    logic_encoder.eval()
    traffic_encoder.eval()
    projection_traffic.eval()
    projection_logic.eval()

    with torch.no_grad():
        traffic_input = traffic_vectors.to(device)  # [B, 1, 1500]
        logic_input = logic_semantics.to(device)

        traffic_embeds = traffic_encoder(traffic_input)
        traffic_embeds = projection_traffic(traffic_embeds)  # [B, 128]
        logic_embeds = logic_encoder(logic_input)
        logic_embeds = projection_logic(logic_embeds)

        logic_embed_norm = F.normalize(logic_embeds, dim=1)
        traffic_embed_norm = F.normalize(traffic_embeds, dim=1)

        # traffic as query
        similarity = traffic_embed_norm @ logic_embed_norm.T  # [N, N]

        top1 = (similarity.argmax(dim=1) == torch.arange(similarity.size(0), device=device)).float().mean()
        top5 = (similarity.topk(5, dim=1).indices == torch.arange(similarity.size(0), device=device).unsqueeze(1)).any(dim=1).float().mean()

        print(f"[VAL] Top-1 acc: {top1.item():.4f}, Top-5 acc: {top5.item():.4f}")
        return top1.item()


def main():
    from torch.nn import DataParallel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_top1 = 0.0
    best_model_path = 'STAR_model_pt/best_STAR_model.pt'

    traffic_encoder = DFEncoder(input_length=5000).to(device)
    logic_encoder = LogicFeatureEncoder().to(device)
    projection_traffic = nn.Linear(512, 256).to(device)
    projection_logic = nn.Linear(512, 256).to(device)

    classifier_for_loss = torch.nn.Linear(256, 395).to(device)

    if torch.cuda.device_count() > 1:
        traffic_encoder = DataParallel(traffic_encoder)
        logic_encoder = DataParallel(logic_encoder)
        projection_traffic = DataParallel(projection_traffic)
        projection_logic = DataParallel(projection_logic)
        classifier_for_loss = DataParallel(classifier_for_loss)

    optimizer = torch.optim.Adam(
        list(traffic_encoder.parameters()) + list(logic_encoder.parameters()) + list(projection_traffic.parameters()) + list(projection_logic.parameters()) + list(classifier_for_loss.parameters()),
        lr=3e-4
    )

    for epoch in range(400):
        loss, acc = train_one_epoch(traffic_encoder, logic_encoder, projection_traffic, projection_logic, classifier_for_loss, dataloader, optimizer, device)
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}, Acc = {acc:.4f}")

        if (epoch + 1) % 5 == 0:
            print(f"\n[INFO] Running validation at epoch {epoch + 1} ...")
            top1 = validate_clip_model(logic_encoder, traffic_encoder, projection_traffic, projection_logic, logic_semantics_val, traffic_vectors_val, device)

            if top1 > best_top1:
                best_top1 = top1
                print(f"[INFO] Saving new best model (top1={top1:.4f}) ...")
                torch.save({
                    'epoch': epoch + 1,
                    'logic_encoder': logic_encoder.state_dict(),
                    'traffic_encoder': traffic_encoder.state_dict(),
                    'projection_logic': projection_logic.state_dict(),
                    'projection_traffic': projection_traffic.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, best_model_path)


if __name__ == '__main__':
    main()
