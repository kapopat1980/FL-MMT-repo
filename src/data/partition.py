"""
Dirichlet-based partitioning of a labeled dataset across simulated
federated clients, used to control non-IID label skew across the
8 simulated clients (Section 3.4).
"""

import numpy as np
from torch.utils.data import Subset


def dirichlet_partition_by_label(dataset, labels: np.ndarray, num_clients: int,
                                  alpha: float = 0.5, seed: int = 42):
    """
    Partitions `dataset` indices into `num_clients` subsets using a
    Dirichlet(alpha) distribution over each class's samples.

    Lower alpha -> more label skew (more non-IID) across clients.

    Args:
        dataset: the full dataset (indexable)
        labels: (N,) array of integer class labels for each dataset item
        num_clients: number of simulated clients
        alpha: Dirichlet concentration parameter
        seed: RNG seed for reproducibility
    Returns:
        list of `Subset` objects, one per client
    """
    rng = np.random.default_rng(seed)
    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_indices = np.where(labels == c)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet(alpha=[alpha] * num_clients)
        # Convert proportions to split points
        split_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, split_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    return [Subset(dataset, idxs) for idxs in client_indices]
