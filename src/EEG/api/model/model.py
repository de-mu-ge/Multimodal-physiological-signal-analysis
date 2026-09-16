import torch
import torch.nn as nn


class EEGModel(nn.Module):

    def __init__(self):
        super().__init__()

        # =========================================================
        # Input:
        #       (B, 2000, 30)
        #
        # 2000 = 时间点
        # 30   = EEG通道
        # =========================================================

        # ---------------------------------------------------------
        # 1. EEG Channel Embedding
        #
        # 每一个时间点：
        #
        #       30 channels
        #           ↓
        #       128-dimensional feature
        #
        # (B, 2000, 30)
        #       ↓
        # (B, 2000, 128)
        # ---------------------------------------------------------

        self.embedding = nn.Sequential(
            nn.Linear(30, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )


        # ---------------------------------------------------------
        # 2. Temporal Downsampling
        #
        # 原来 2000 个时间点
        # ↓
        # 250 个 token
        #
        # 不碰 EEG channel 维度
        #
        # (B, 2000, 128)
        #       ↓
        # (B, 250, 128)
        # ---------------------------------------------------------

        self.temporal_downsample = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=8,
                stride=8
            ),
            nn.BatchNorm1d(128),
            nn.GELU()
        )


        # ---------------------------------------------------------
        # 3. Positional Encoding
        #
        # Transformer 必须知道时间顺序
        # ---------------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.randn(1, 250, 128) * 0.02
        )


        # ---------------------------------------------------------
        # 4. Transformer Encoder
        #
        # sequence:
        #
        # (B, 250, 128)
        #
        # 250 = 时间 token
        # 128 = embedding
        # ---------------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=3
        )


        # ---------------------------------------------------------
        # 5. Classification
        # ---------------------------------------------------------

        self.norm = nn.LayerNorm(128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )


    def forward(self, x):

        # =========================================================
        # x
        #
        # (B, 2000, 30)
        # =========================================================

        # ---------------------------------------------------------
        # Channel embedding
        # ---------------------------------------------------------

        x = self.embedding(x)

        # (B, 2000, 128)


        # ---------------------------------------------------------
        # Temporal downsampling
        # ---------------------------------------------------------

        # Conv1d 要求：
        #
        # (B, C, L)
        #
        x = x.transpose(1, 2)

        # (B, 128, 2000)

        x = self.temporal_downsample(x)

        # (B, 128, 250)


        # Transformer 要求：
        #
        # (B, L, C)
        #
        x = x.transpose(1, 2)

        # (B, 250, 128)


        # ---------------------------------------------------------
        # Position
        # ---------------------------------------------------------

        x = x + self.position_embedding


        # ---------------------------------------------------------
        # Transformer
        # ---------------------------------------------------------

        x = self.transformer(x)

        # (B, 250, 128)


        # ---------------------------------------------------------
        # Global Average Pooling
        # ---------------------------------------------------------

        x = x.mean(dim=1)

        # (B, 128)


        # ---------------------------------------------------------
        # Classification
        # ---------------------------------------------------------

        x = self.norm(x)

        x = self.classifier(x)

        # (B, 2)

        return x