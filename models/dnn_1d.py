"""
dnn_1d.py
---------
Simple 2-layer 1D fully connected DNN as per benchmark paper.
"""

import torch
import torch.nn as nn


class DNN1D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_ch = cfg.input_channels
        h1, h2 = cfg.hidden_dims
        num_classes = cfg.num_classes

        self.net = nn.Sequential(
            nn.Linear(in_ch, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

