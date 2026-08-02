"""
FedProx client-side training and server-side aggregation.

Implements the proximal-term local objective from Li et al., 2020
("Federated Optimization in Heterogeneous Networks", MLSys), used here
to stabilise training across the 8 simulated, non-IID clients described
in Section 3.4 of the manuscript.
"""

import copy
import torch


def fedprox_local_update(model, global_state_dict, dataloader, optimizer,
                          loss_fn, mu: float = 0.01, local_epochs: int = 1, device="cuda"):
    """
    Runs local FedProx training for one client.

    Args:
        model: local copy of the model, already loaded with global_state_dict
        global_state_dict: state dict of the global model (kept fixed during local steps)
        dataloader: client's local DataLoader
        optimizer: local optimizer (may be wrapped by Opacus for DP-SGD, see dp_sgd.py)
        loss_fn: task loss function
        mu: proximal term coefficient
        local_epochs: number of local epochs per communication round
    Returns:
        Updated local state_dict
    """
    model.train()
    global_params = {k: v.clone().detach().to(device) for k, v in global_state_dict.items()}

    for _ in range(local_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            task_loss = loss_fn(outputs, batch)

            # Proximal term: penalize drift from the global model
            prox_term = 0.0
            for name, param in model.named_parameters():
                if name in global_params:
                    prox_term += torch.sum((param - global_params[name]) ** 2)
            loss = task_loss + (mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

    return {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}


def fedavg_aggregate(client_state_dicts, client_weights=None):
    """
    Weighted federated averaging of client state dicts.

    Args:
        client_state_dicts: list of state_dict objects from each client
        client_weights: optional list of weights (e.g. proportional to local
                        dataset size); defaults to uniform averaging.
    Returns:
        Aggregated global state_dict
    """
    n = len(client_state_dicts)
    if client_weights is None:
        client_weights = [1.0 / n] * n
    else:
        total = sum(client_weights)
        client_weights = [w / total for w in client_weights]

    agg_state = copy.deepcopy(client_state_dicts[0])
    for key in agg_state:
        agg_state[key] = sum(w * sd[key].float() for w, sd in zip(client_weights, client_state_dicts))

    return agg_state
