"""
Hierarchical Cross-Modal Attention (HCMA) encoder.

Fuses four modality embeddings (CT, histopathology, genomic, clinical)
through stacked bidirectional cross-attention blocks, as described in
Section 3.2 of the manuscript.
"""

import torch
import torch.nn as nn


class CrossModalAttentionBlock(nn.Module):
    """Single bidirectional cross-attention block between two modality streams."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_a_to_b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_b_to_a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.ffn_a = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ffn_b = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor):
        attn_a, _ = self.attn_a_to_b(x_a, x_b, x_b)
        x_a = self.norm_a(x_a + self.dropout(attn_a))
        x_a = self.norm_a(x_a + self.dropout(self.ffn_a(x_a)))

        attn_b, _ = self.attn_b_to_a(x_b, x_a, x_a)
        x_b = self.norm_b(x_b + self.dropout(attn_b))
        x_b = self.norm_b(x_b + self.dropout(self.ffn_b(x_b)))

        return x_a, x_b


class HCMAEncoder(nn.Module):
    """
    Hierarchical Cross-Modal Attention encoder.

    Iteratively applies cross-attention across all modality pairs for
    `num_layers` rounds, then aggregates via modality-dropout-aware pooling
    to produce a single fused representation.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, num_layers: int = 4,
                 modalities=("ct", "histo", "genomic", "clinical"), dropout: float = 0.1):
        super().__init__()
        self.modalities = list(modalities)
        self.num_layers = num_layers

        # One cross-attention block per unordered modality pair, repeated per layer
        pairs = [(i, j) for i in range(len(self.modalities)) for j in range(i + 1, len(self.modalities))]
        self.pairs = pairs
        self.blocks = nn.ModuleList([
            nn.ModuleList([CrossModalAttentionBlock(dim, num_heads, dropout) for _ in pairs])
            for _ in range(num_layers)
        ])

        self.pool_proj = nn.Linear(dim * len(self.modalities), dim)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, modality_embeddings: dict, modality_mask: dict = None):
        """
        Args:
            modality_embeddings: dict mapping modality name -> tensor (B, T, dim)
            modality_mask: dict mapping modality name -> bool, True if present.
                           Missing modalities are zero-filled upstream (see
                           src/training/collate.py) and excluded from pooling.
        Returns:
            fused: tensor (B, dim) — fused multi-modal representation
        """
        streams = {m: modality_embeddings[m] for m in self.modalities}

        for layer_idx in range(self.num_layers):
            block_set = self.blocks[layer_idx]
            for (i, j), block in zip(self.pairs, block_set):
                m_i, m_j = self.modalities[i], self.modalities[j]
                streams[m_i], streams[m_j] = block(streams[m_i], streams[m_j])

        pooled = []
        for m in self.modalities:
            token_pooled = streams[m].mean(dim=1)  # (B, dim)
            if modality_mask is not None and m in modality_mask:
                mask = modality_mask[m].float().unsqueeze(-1)  # (B, 1)
                token_pooled = token_pooled * mask
            pooled.append(token_pooled)

        fused = torch.cat(pooled, dim=-1)  # (B, dim * n_modalities)
        fused = self.pool_proj(fused)
        return self.final_norm(fused)
