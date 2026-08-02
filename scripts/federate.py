"""
Entry point for simulated federated training with FedProx + DP-SGD across
8 clients (Section 3.4 / 4.3 of the manuscript).

Usage:
    python scripts/federate.py --config configs/federated_dp.yaml
"""

import argparse
import copy
import yaml
import torch

from src.models.hcma_encoder import HCMAEncoder
from src.models.task_heads import FLMMTTaskHeads
from src.federated.fedprox import fedprox_local_update, fedavg_aggregate
from src.federated.dp_sgd import attach_dp_sgd, get_privacy_spent
from src.data.dataset import build_federated_client_loaders
from src.training.losses import multi_task_loss


class FLMMTModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hcma = HCMAEncoder(**cfg["model"]["hcma"])
        self.task_heads = FLMMTTaskHeads(fused_dim=cfg["model"]["hcma"]["dim"], **cfg["model"]["task_heads"])

    def forward(self, batch):
        fused = self.hcma(batch["modality_embeddings"], batch.get("modality_mask"))
        return self.task_heads(fused)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["training"]["seed"])
    device = cfg["training"]["device"]

    global_model = FLMMTModel(cfg).to(device)
    client_loaders = build_federated_client_loaders(cfg)

    for round_idx in range(cfg["federated"]["communication_rounds"]):
        client_states = []
        client_weights = []

        for client_id, loader in enumerate(client_loaders):
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=cfg["optimizer"]["lr"],
                                           weight_decay=cfg["optimizer"]["weight_decay"])

            local_model, optimizer, private_loader, privacy_engine = attach_dp_sgd(
                local_model, optimizer, loader,
                target_epsilon=cfg["privacy"]["target_epsilon"],
                target_delta=cfg["privacy"]["target_delta"],
                epochs=cfg["federated"]["communication_rounds"] * cfg["federated"]["local_epochs"],
                max_grad_norm=cfg["privacy"]["max_grad_norm"],
            )

            updated_state = fedprox_local_update(
                local_model, global_model.state_dict(), private_loader, optimizer,
                loss_fn=multi_task_loss, mu=cfg["federated"]["mu"],
                local_epochs=cfg["federated"]["local_epochs"], device=device,
            )

            client_states.append(updated_state)
            client_weights.append(len(loader.dataset))

        aggregated_state = fedavg_aggregate(client_states, client_weights)
        global_model.load_state_dict(aggregated_state)

        eps_spent = get_privacy_spent(privacy_engine, cfg["privacy"]["target_delta"])
        print(f"[round {round_idx + 1}] aggregated global model | "
              f"privacy spent so far: epsilon={eps_spent:.3f}")

    torch.save(global_model.state_dict(), "checkpoints/fl_mmt_federated_best.pt")


if __name__ == "__main__":
    main()
