"""
Multi-task loss combining classification cross-entropy, risk-stratification
cross-entropy, and a discretized-time survival loss (Dynamic-DeepHit style
combination of log-likelihood and ranking loss).
"""

import torch
import torch.nn.functional as F


def survival_loss(survival_logits, time_bin, event_observed, rank_weight: float = 0.1):
    """
    Discretized-time survival loss: negative log-likelihood of the observed
    event/time bin plus a pairwise concordance ranking term, following
    Lee et al. (2020), "Dynamic-DeepHit".
    """
    survival_probs = torch.softmax(survival_logits, dim=-1)
    log_likelihood = -torch.log(
        torch.gather(survival_probs, 1, time_bin.unsqueeze(1)).squeeze(1) + 1e-8
    )
    nll = (log_likelihood * event_observed.float()).mean()

    # Simplified pairwise ranking term encouraging concordance between
    # predicted risk ordering and observed event times.
    cdf = torch.cumsum(survival_probs, dim=-1)
    risk_score = 1.0 - cdf.gather(1, time_bin.unsqueeze(1)).squeeze(1)
    rank_loss = 0.0
    n = risk_score.size(0)
    if n > 1:
        diff_time = time_bin.unsqueeze(0) - time_bin.unsqueeze(1)
        diff_risk = risk_score.unsqueeze(0) - risk_score.unsqueeze(1)
        mask = (diff_time > 0).float()
        rank_loss = (mask * torch.relu(-diff_risk)).sum() / (mask.sum() + 1e-8)

    return nll + rank_weight * rank_loss


def multi_task_loss(outputs, batch, weights=(1.0, 1.0, 1.0)):
    """
    Combines classification, risk-stratification, and survival losses.

    Args:
        outputs: dict with keys classification_logits, risk_logits, survival_logits
        batch: dict with keys label, risk_label, survival_time_bin, event_observed
        weights: (w_cls, w_risk, w_surv) loss-combination weights
    """
    w_cls, w_risk, w_surv = weights

    cls_loss = F.cross_entropy(outputs["classification_logits"], batch["label"])
    risk_loss = F.cross_entropy(outputs["risk_logits"], batch["risk_label"])
    surv_loss = survival_loss(outputs["survival_logits"], batch["survival_time_bin"], batch["event_observed"])

    return w_cls * cls_loss + w_risk * risk_loss + w_surv * surv_loss
