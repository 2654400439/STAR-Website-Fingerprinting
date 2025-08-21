import torch
import torch.nn as nn


class LogicFeatureEncoder(nn.Module):
    def __init__(self, d_model=128, max_len=80):
        super().__init__()
        self.max_len = max_len

        self.linear_proj = nn.Linear(4, d_model)

        self.embed_type = nn.Embedding(11, d_model)
        self.embed_ip = nn.Embedding(21, d_model)
        self.embed_proto = nn.Embedding(4, d_model)

        self.support_proj = nn.Linear(1, d_model)

        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.output_proj = nn.Linear(d_model, 512)  # output embedding size

    def forward(self, x):  # x: [B, 80, 8]
        x = x.float()

        cont_feat = torch.stack([x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 7]], dim=-1)  # [B, 80, 4]
        type_ids = x[:, :, 3].long()
        ip_ids = x[:, :, 4].long().clamp(max=20)
        support_h3 = x[:, :, 5].unsqueeze(-1)  # [B, 80, 1]
        proto_ids = x[:, :, 6].long()

        cont_embed = self.linear_proj(cont_feat)             # [B, 80, d_model]
        type_embed = self.embed_type(type_ids)               # [B, 80, d_model]
        ip_embed = self.embed_ip(ip_ids)
        proto_embed = self.embed_proto(proto_ids)
        support_embed = self.support_proj(support_h3)        # [B, 80, d_model]

        x_embed = cont_embed + type_embed + ip_embed + proto_embed + support_embed  # [B, 80, d_model]

        x_embed = x_embed + self.pos_embed  # [B, 80, d_model]

        mask = (x.sum(dim=-1) == 0)  # [B, 80]

        # transformer encoder
        x_encoded = self.encoder(x_embed, src_key_padding_mask=mask)

        x_out = x_encoded.masked_fill(mask.unsqueeze(-1), 0.0)
        valid_counts = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        x_out = x_out.sum(dim=1) / valid_counts

        return self.output_proj(x_out)  # [B, 512]