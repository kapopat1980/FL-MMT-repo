"""
Entry point for centralised (non-federated) training of FL-MMT.
Used to produce the centralised baseline reported in Table 6.

Usage:
    python scripts/train.py --config configs/centralised.yaml
"""

import argparse
import yaml
import torch

from src.models.hcma_encoder import HCMAEncoder
from src.models.task_heads import FLMMTTaskHeads
from src.training.trainer import Trainer
from src.data.dataset import build_dataloaders


def build_model(cfg):
    hcma = HCMAEncoder(**cfg["model"]["hcma"])
    heads = FLMMTTaskHeads(fused_dim=cfg["model"]["hcma"]["dim"], **cfg["model"]["task_heads"])
    return hcma, heads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])
    device = cfg["training"]["device"]

    hcma, heads = build_model(cfg)
    hcma.to(device)
    heads.to(device)

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    trainer = Trainer(hcma, heads, cfg, device=device)
    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])
    trainer.evaluate(test_loader)


if __name__ == "__main__":
    main()
