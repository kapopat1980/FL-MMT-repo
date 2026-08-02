"""
Centralised training loop for the FL-MMT model, used to produce the
centralised (non-federated) baseline reported in Table 6.
"""

import os
import torch
from tqdm import tqdm

from src.training.losses import multi_task_loss


class Trainer:
    def __init__(self, hcma, task_heads, cfg, device="cuda"):
        self.hcma = hcma
        self.task_heads = task_heads
        self.cfg = cfg
        self.device = device

        params = list(self.hcma.parameters()) + list(self.task_heads.parameters())
        self.optimizer = torch.optim.AdamW(
            params, lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"]
        )

    def forward(self, batch):
        modality_embeddings = {k: v.to(self.device) for k, v in batch["modality_embeddings"].items()}
        modality_mask = {k: v.to(self.device) for k, v in batch["modality_mask"].items()}
        fused = self.hcma(modality_embeddings, modality_mask)
        return self.task_heads(fused)

    def fit(self, train_loader, val_loader, epochs: int):
        best_val_loss = float("inf")
        os.makedirs("checkpoints", exist_ok=True)

        for epoch in range(epochs):
            self.hcma.train()
            self.task_heads.train()
            running_loss = 0.0

            for batch in tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}"):
                self.optimizer.zero_grad()
                outputs = self.forward(batch)
                loss = multi_task_loss(outputs, batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            val_loss = self._validate(val_loader)
            print(f"epoch {epoch + 1}: train_loss={running_loss / len(train_loader):.4f}, "
                  f"val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {"hcma": self.hcma.state_dict(), "task_heads": self.task_heads.state_dict()},
                    "checkpoints/fl_mmt_best.pt",
                )

    @torch.no_grad()
    def _validate(self, val_loader):
        self.hcma.eval()
        self.task_heads.eval()
        total_loss = 0.0
        for batch in val_loader:
            outputs = self.forward(batch)
            loss = multi_task_loss(outputs, batch)
            total_loss += loss.item()
        return total_loss / len(val_loader)

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.hcma.eval()
        self.task_heads.eval()
        # See src/evaluation/ for metric computation (macro-AUC, ECE, C-index, etc.)
        print("Run scripts/evaluate.py against the saved checkpoint for full metrics.")
