"""Evaluation metrics for sector tendency predictions."""

import numpy as np


def directional_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of predictions where sign matches target sign.

    Both arrays should be 1D. Values at 0 are excluded.
    """
    mask = (targets != 0) & (predictions != 0)
    if mask.sum() == 0:
        return 0.0
    pred_sign = np.sign(predictions[mask])
    target_sign = np.sign(targets[mask])
    return (pred_sign == target_sign).mean()


def tendency_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error of tendency scores."""
    mask = ~np.isnan(targets) & ~np.isnan(predictions)
    if mask.sum() == 0:
        return float("nan")
    return ((predictions[mask] - targets[mask]) ** 2).mean()


def calibration_error(
    predictions: np.ndarray,
    targets: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error.

    Groups predictions by confidence, checks if accuracy matches confidence.
    """
    correct = np.sign(predictions) == np.sign(targets)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(predictions)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / total * abs(bin_acc - bin_conf)

    return ece


def momentum_baseline(prices: np.ndarray, horizon: int) -> np.ndarray:
    """Simple momentum baseline: last period's return predicts next period.

    This is the minimum bar our model must beat.
    """
    past_returns = np.diff(prices, axis=0) / prices[:-1]
    # Use last `horizon` days' return as prediction for next `horizon` days
    if len(past_returns) < horizon:
        return np.zeros(len(prices))
    momentum = np.convolve(past_returns, np.ones(horizon) / horizon, mode="valid")
    # Pad to match original length
    pad = np.full(len(prices) - len(momentum), np.nan)
    return np.concatenate([pad, momentum])


def evaluate_sector(
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    confidences: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate all sectors, return metrics per sector.

    Args:
        predictions: sector_token -> 1D array of tendency predictions
        targets: sector_token -> 1D array of actual tendency values
        confidences: optional, sector_token -> 1D array of confidence scores
    """
    results = {}
    for sector in predictions:
        if sector not in targets:
            continue
        pred = predictions[sector]
        tgt = targets[sector]

        metrics = {
            "directional_accuracy": directional_accuracy(pred, tgt),
            "mse": tendency_mse(pred, tgt),
        }

        if confidences and sector in confidences:
            metrics["calibration_error"] = calibration_error(pred, tgt, confidences[sector])

        results[sector] = metrics

    return results
