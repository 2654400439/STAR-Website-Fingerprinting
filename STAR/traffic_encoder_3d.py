import torch
import torch.nn as nn
import torch.nn.functional as F

class DFEncoder(nn.Module):
    def __init__(self, input_length=5000, flow_id_vocab_size=256, flow_id_emb_dim=8, proto_num=3):
        super(DFEncoder, self).__init__()

        self.flow_id_embed = nn.Embedding(flow_id_vocab_size, flow_id_emb_dim, padding_idx=0)
        self.proto_num = proto_num

        # input = pkt_len + flow_id_emb_dim + proto_num
        total_input_channels = 1 + flow_id_emb_dim + proto_num

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(total_input_channels, 32, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
            nn.Conv1d(32, 32, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(32),
            nn.ELU(alpha=1.0),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
            nn.Dropout(0.1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
            nn.Dropout(0.1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
            nn.Dropout(0.1)
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=8, stride=1, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=4, padding=2),
            nn.Dropout(0.1)
        )

        self._output_dim = self._get_flattened_size(input_length)

        self.encoder_fc = nn.Sequential(
            nn.Linear(self._output_dim, 512)
        )

    def _get_flattened_size(self, input_length):
        x = torch.zeros(1, 3, input_length)  # placeholder input
        x = self._preprocess_input(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        return x.view(1, -1).shape[1]

    def _preprocess_input(self, x):
        length_feat = x[:, 0, :]                  # [B, L]
        flow_ids = x[:, 1, :].long().clamp(min=0, max=255)              # [B, L]
        proto_ids = x[:, 2, :].long()             # [B, L]

        ch0 = length_feat.unsqueeze(1)            # [B, 1, L]
        ch1 = self.flow_id_embed(flow_ids)       # [B, L, D] -> [B, D, L]
        ch1 = ch1.permute(0, 2, 1)
        ch2 = F.one_hot(proto_ids, num_classes=self.proto_num).float().permute(0, 2, 1)  # [B, 3, L]

        return torch.cat([ch0, ch1, ch2], dim=1)  # [B, C_total, L]

    def forward(self, x):  # x: [B, 3, L]
        x = self._preprocess_input(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)
        return x
