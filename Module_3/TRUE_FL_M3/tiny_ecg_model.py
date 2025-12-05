# PART-3/src/tiny_ecg_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyECG(nn.Module):
    """
    Very small 1D CNN for ECG windows.
    Input: [B, 1, L] (L ≈ 500)
    Output: [B, num_classes]
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        # x: [B, 1, L]
        x = self.net(x)          # [B, 16, 1]
        x = x.squeeze(-1)        # [B, 16]
        return self.head(x)      # [B, C]
