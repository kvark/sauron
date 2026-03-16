"""Custom loss functions for Sauron.

Primary loss: Pinball (quantile) loss for probabilistic tendency predictions.
Additional losses: direction accuracy, confidence calibration.
"""

import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    """Quantile regression loss (pinball loss).

    For quantile q: L = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    """

    def __init__(self, quantiles: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute pinball loss.

        Args:
            predictions: (batch, num_quantiles) predicted quantile values
            targets: (batch,) actual values

        Returns:
            scalar loss
        """
        targets = targets.unsqueeze(-1).expand_as(predictions)
        errors = targets - predictions
        losses = torch.zeros_like(errors)

        for i, q in enumerate(self.quantiles):
            losses[:, i] = torch.max(q * errors[:, i], (q - 1) * errors[:, i])

        return losses.mean()


class TendencyLoss(nn.Module):
    """Combined loss for sector tendency prediction.

    Components:
    1. MSE on tendency value (primary signal)
    2. Direction accuracy penalty (extra loss for wrong sign)
    3. Confidence calibration (confident predictions should be accurate)
    """

    def __init__(self, direction_weight: float = 0.3, confidence_weight: float = 0.1):
        super().__init__()
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred_tendency: torch.Tensor,
        pred_confidence: torch.Tensor,
        target_tendency: torch.Tensor,
    ) -> torch.Tensor:
        # MSE on tendency value
        mse_loss = self.mse(pred_tendency, target_tendency)

        # Direction penalty: extra loss when predicted sign != actual sign
        pred_sign = torch.sign(pred_tendency)
        target_sign = torch.sign(target_tendency)
        direction_wrong = (pred_sign != target_sign).float()
        direction_loss = direction_wrong.mean()

        # Confidence calibration: high confidence + wrong = high penalty
        error = (pred_tendency - target_tendency).abs()
        confidence_loss = (pred_confidence * error).mean()

        return mse_loss + self.direction_weight * direction_loss + self.confidence_weight * confidence_loss
