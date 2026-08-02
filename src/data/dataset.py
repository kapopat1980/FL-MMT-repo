"""
Dataset loaders for the five public datasets used in FL-MMT.

Each modality is preprocessed independently and cached; a patient-level
manifest joins across modalities where available and applies zero-filling
for missing modalities (see collate_fn below), consistent with the
modality-dropout robustness strategy described in Section 3.5.

NOTE: raw imaging data is never included in this repository. See
docs/dataset_setup.md for download and preprocessing instructions for:
  - LC25000 (Kaggle / Borkowski et al.)
  - TCGA-LUAD / TCGA-LUSC (NCI GDC Data Portal)
  - LIDC-IDRI (The Cancer Imaging Archive)
  - NLST (The Cancer Imaging Archive)
  - IQ-OTH/NCCD (IEEE DataPort)
"""

import torch
from torch.utils.data import Dataset, DataLoader


class FLMMTDataset(Dataset):
    """
    Patient-level multi-modal dataset. Each item returns a dict of
    per-modality tensors (some possibly absent, handled via masking)
    plus classification/risk/survival labels.
    """

    def __init__(self, manifest_path: str, split: str, modalities=("ct", "histo", "genomic", "clinical")):
        self.manifest_path = manifest_path
        self.split = split
        self.modalities = modalities
        self.records = self._load_manifest(manifest_path, split)

    @staticmethod
    def _load_manifest(manifest_path, split):
        # Implementation loads a per-patient CSV/JSON manifest built by the
        # preprocessing scripts in docs/dataset_setup.md, filtered to `split`.
        raise NotImplementedError(
            "Manifest loading is dataset-specific and requires the preprocessed "
            "manifests produced by the scripts in docs/dataset_setup.md. "
            "This is a structural placeholder for the public code release."
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        item = {"modality_embeddings": {}, "modality_mask": {}}
        for modality in self.modalities:
            if modality in record and record[modality] is not None:
                item["modality_embeddings"][modality] = record[modality]
                item["modality_mask"][modality] = torch.tensor(1.0)
            else:
                item["modality_embeddings"][modality] = torch.zeros(record["embedding_dim"])
                item["modality_mask"][modality] = torch.tensor(0.0)

        item["label"] = record["label"]
        item["risk_label"] = record["risk_label"]
        item["survival_time_bin"] = record["survival_time_bin"]
        item["event_observed"] = record["event_observed"]
        return item


def build_dataloaders(cfg):
    train_ds = FLMMTDataset(cfg["data"].get("manifest_path", "data/manifest.csv"), split="train")
    val_ds = FLMMTDataset(cfg["data"].get("manifest_path", "data/manifest.csv"), split="val")
    test_ds = FLMMTDataset(cfg["data"].get("manifest_path", "data/manifest.csv"), split="test")

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def build_federated_client_loaders(cfg):
    """
    Partitions the training set across `federated.num_clients` clients using
    a Dirichlet distribution over labels (alpha = federated.dirichlet_alpha)
    to simulate non-IID clinical-site heterogeneity, as described in
    Section 3.4.
    """
    num_clients = cfg["federated"]["num_clients"]
    batch_size = cfg["training"]["batch_size"]

    full_train_ds = FLMMTDataset(cfg["data"].get("manifest_path", "data/manifest.csv"), split="train")

    # Placeholder: actual Dirichlet partitioning implemented in
    # src/data/partition.py (dirichlet_partition_by_label).
    raise NotImplementedError(
        "Client partitioning requires the preprocessed manifest and label "
        "distribution; see src/data/partition.py and docs/dataset_setup.md."
    )
