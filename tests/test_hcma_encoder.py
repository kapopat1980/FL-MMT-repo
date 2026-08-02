import torch
from src.models.hcma_encoder import HCMAEncoder


def test_hcma_forward_shape():
    dim = 64
    batch_size = 4
    seq_len = 10
    modalities = ("ct", "histo", "genomic", "clinical")

    encoder = HCMAEncoder(dim=dim, num_heads=4, num_layers=2, modalities=modalities)

    embeddings = {m: torch.randn(batch_size, seq_len, dim) for m in modalities}
    mask = {m: torch.ones(batch_size) for m in modalities}

    fused = encoder(embeddings, mask)

    assert fused.shape == (batch_size, dim)


def test_hcma_handles_missing_modality():
    dim = 64
    batch_size = 4
    seq_len = 10
    modalities = ("ct", "histo", "genomic", "clinical")

    encoder = HCMAEncoder(dim=dim, num_heads=4, num_layers=2, modalities=modalities)

    embeddings = {m: torch.randn(batch_size, seq_len, dim) for m in modalities}
    mask = {m: torch.ones(batch_size) for m in modalities}
    mask["genomic"] = torch.zeros(batch_size)  # simulate missing modality

    fused = encoder(embeddings, mask)

    assert fused.shape == (batch_size, dim)
    assert torch.isfinite(fused).all()
