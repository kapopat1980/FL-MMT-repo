"""
Task heads for FL-MMT: classification, risk stratification, and survival
(Dynamic-DeepHit-style discretized survival head), sharing the fused
HCMA representation. Dropout is kept active at inference time to support
Monte Carlo Dropout uncertainty estimation (see src/evaluation/calibration.py).
"""

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FLMMTTaskHeads(nn.Module):
    """
    Bundles the three task heads used in the manuscript:
      - classification: 6-class subtype classification (softmax logits)
      - risk: binary/ordinal risk stratification head
      - survival: discretized-time survival head (Dynamic-DeepHit style)
    """

    def __init__(self, fused_dim: int = 512, num_classes: int = 6,
                 num_risk_classes: int = 3, num_time_bins: int = 20, dropout: float = 0.3):
        super().__init__()
        self.classification_head = MLPHead(fused_dim, num_classes, dropout=dropout)
        self.risk_head = MLPHead(fused_dim, num_risk_classes, dropout=dropout)
        self.survival_head = MLPHead(fused_dim, num_time_bins, dropout=dropout)

    def forward(self, fused_repr: torch.Tensor):
        return {
            "classification_logits": self.classification_head(fused_repr),
            "risk_logits": self.risk_head(fused_repr),
            "survival_logits": self.survival_head(fused_repr),
        }

    def enable_mc_dropout(self):
        """Keep dropout layers active during eval() for MC Dropout sampling."""
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()
