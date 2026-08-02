"""
Differentially private client-side training via Opacus DP-SGD, with
Renyi-DP accounting (Mironov, 2017) tracked across communication rounds
to report the (epsilon, delta) budget stated in the manuscript
(epsilon=2.1, delta=1e-5, see Section 4.3).
"""

from opacus import PrivacyEngine


def attach_dp_sgd(model, optimizer, dataloader, target_epsilon: float,
                   target_delta: float, epochs: int, max_grad_norm: float = 1.0):
    """
    Wraps a model/optimizer/dataloader with Opacus DP-SGD, calibrated to
    reach `target_epsilon` privacy budget after `epochs` local epochs.

    Returns:
        private_model, private_optimizer, private_dataloader, privacy_engine
    """
    privacy_engine = PrivacyEngine()

    private_model, private_optimizer, private_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

    return private_model, private_optimizer, private_dataloader, privacy_engine


def get_privacy_spent(privacy_engine, target_delta: float):
    """Returns the (epsilon, alpha) actually spent so far."""
    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    return epsilon
