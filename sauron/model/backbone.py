"""Layer 1: Foundation model backbone wrappers for Chronos-2 and MOIRAI-2.

These models handle the core forecasting task. Our domain layers (event encoder,
sector graph, scenario engine) sit on top to add geopolitical intelligence.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class ForecastBackbone(ABC):
    """Abstract base for time-series foundation model backends."""

    @abstractmethod
    def predict(
        self,
        context: np.ndarray,
        horizon: int,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> dict[str, np.ndarray]:
        """Generate probabilistic forecasts.

        Args:
            context: (batch, seq_len, num_series) historical values
            horizon: number of steps to forecast
            quantiles: quantile levels for probabilistic output

        Returns:
            dict with 'median', 'lower', 'upper' arrays of shape (batch, horizon, num_series)
        """
        ...

    @abstractmethod
    def encode(self, context: np.ndarray) -> np.ndarray:
        """Extract learned representations from the context window.

        Args:
            context: (batch, seq_len, num_series) historical values

        Returns:
            (batch, embedding_dim) representation for downstream layers
        """
        ...


class Chronos2Backbone(ForecastBackbone):
    """Amazon Chronos-2 wrapper.

    Key capability: group attention over multiple related time series with covariates.
    We feed sector time series as groups (e.g., GDP + trade + energy for a country-sector pair).
    """

    def __init__(self, model_id: str = "amazon/chronos-bolt-base", device: str = "cpu"):
        self.device = device
        self.model_id = model_id
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from chronos import ChronosPipeline

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
        return self._pipeline

    def predict(
        self,
        context: np.ndarray,
        horizon: int,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> dict[str, np.ndarray]:
        """Generate forecasts using Chronos-2 group attention.

        Context can be multivariate — Chronos-2 handles cross-series relationships
        via its group attention mechanism.
        """
        # Chronos expects list of 1D tensors or a 2D tensor (batch x time)
        if context.ndim == 3:
            # (batch, time, series) -> process each series group
            batch, seq_len, num_series = context.shape
            all_forecasts = []

            for b in range(batch):
                # Feed all series for this sample as a group
                series_list = [
                    torch.tensor(context[b, :, s], dtype=torch.float32)
                    for s in range(num_series)
                ]
                forecast = self.pipeline.predict(
                    series_list,
                    prediction_length=horizon,
                    num_samples=200,
                )
                all_forecasts.append(forecast)

            # Aggregate quantiles across samples
            result = {"quantiles": {}}
            for q in quantiles:
                q_values = []
                for forecast in all_forecasts:
                    # forecast shape: (num_series, num_samples, horizon)
                    q_val = np.quantile(forecast.numpy(), q, axis=1)
                    q_values.append(q_val)
                result["quantiles"][q] = np.stack(q_values)

            result["median"] = result["quantiles"][0.5]
            return result

        elif context.ndim == 2:
            # (batch, time) -> univariate per sample
            tensors = [torch.tensor(context[i], dtype=torch.float32) for i in range(len(context))]
            forecast = self.pipeline.predict(
                tensors,
                prediction_length=horizon,
                num_samples=200,
            )
            result = {}
            for q in quantiles:
                result[f"q{q}"] = np.quantile(forecast.numpy(), q, axis=1)
            result["median"] = np.median(forecast.numpy(), axis=1)
            return result

        else:
            raise ValueError(f"Expected 2D or 3D context, got {context.ndim}D")

    def encode(self, context: np.ndarray) -> np.ndarray:
        """Extract Chronos-2 internal representations.

        Uses the encoder output before the forecasting head as features
        for our downstream domain layers.
        """
        # Access the underlying transformer encoder
        model = self.pipeline.model
        if context.ndim == 2:
            context = context[:, :, np.newaxis]

        batch, seq_len, num_series = context.shape
        embeddings = []

        with torch.no_grad():
            for b in range(batch):
                # Tokenize and encode
                series = torch.tensor(context[b, :, 0], dtype=torch.float32).unsqueeze(0)
                # Get encoder hidden states
                tokens = self.pipeline.tokenizer(series)
                enc_out = model.encoder(tokens).last_hidden_state
                # Mean pool over time dimension
                emb = enc_out.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(emb)

        return np.stack(embeddings)


class Moirai2Backbone(ForecastBackbone):
    """Salesforce MOIRAI-2 wrapper.

    Decoder-only, 11M params, fast inference. Used as ensemble member
    and lightweight baseline for rapid iteration.
    """

    def __init__(self, model_id: str = "Salesforce/moirai-2.0-R-small", device: str = "cpu"):
        self.device = device
        self.model_id = model_id
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from uni2ts.model.moirai import MoiraiForecast

            self._model = MoiraiForecast.load_from_checkpoint(
                self.model_id,
                map_location=self.device,
            )
        return self._model

    def predict(
        self,
        context: np.ndarray,
        horizon: int,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> dict[str, np.ndarray]:
        """Generate forecasts using MOIRAI-2 decoder-only architecture."""
        forecast = self.model.forecast(
            context=torch.tensor(context, dtype=torch.float32),
            prediction_length=horizon,
        )

        result = {}
        for q in quantiles:
            result[f"q{q}"] = np.quantile(forecast.numpy(), q, axis=-1)
        result["median"] = np.median(forecast.numpy(), axis=-1)
        return result

    def encode(self, context: np.ndarray) -> np.ndarray:
        """Extract MOIRAI-2 representations."""
        with torch.no_grad():
            hidden = self.model.encode(torch.tensor(context, dtype=torch.float32))
            return hidden.mean(dim=1).cpu().numpy()


class EnsembleBackbone(ForecastBackbone):
    """Weighted ensemble of multiple foundation model backends."""

    def __init__(
        self,
        backends: list[tuple[ForecastBackbone, float]],
    ):
        """Args: list of (backbone, weight) tuples. Weights should sum to 1."""
        self.backends = backends
        total = sum(w for _, w in backends)
        self.backends = [(b, w / total) for b, w in backends]

    def predict(
        self,
        context: np.ndarray,
        horizon: int,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> dict[str, np.ndarray]:
        weighted_medians = None

        for backbone, weight in self.backends:
            result = backbone.predict(context, horizon, quantiles)
            median = result["median"]
            if weighted_medians is None:
                weighted_medians = weight * median
            else:
                weighted_medians = weighted_medians + weight * median

        return {"median": weighted_medians}

    def encode(self, context: np.ndarray) -> np.ndarray:
        embeddings = []
        for backbone, _ in self.backends:
            embeddings.append(backbone.encode(context))
        return np.concatenate(embeddings, axis=-1)
