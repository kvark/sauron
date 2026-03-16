"""Training loop for the Sauron geo-economic model."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sauron.model.losses import TendencyLoss
from sauron.sectors import SECTORS
from sauron.training.config import TrainingConfig


class SauronDatasetTorch(Dataset):
    """PyTorch Dataset wrapper for pipeline output."""

    def __init__(self, samples: list[dict]):
        self.samples = samples
        self.sector_names = list(SECTORS.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        features = torch.tensor(s["features"], dtype=torch.float32)
        mask = torch.tensor(s["mask"], dtype=torch.float32)

        # Build label tensor: one value per sector
        labels = torch.full((len(self.sector_names),), float("nan"))
        for i, sector in enumerate(self.sector_names):
            if sector in s["labels"]:
                labels[i] = s["labels"][sector]

        return {"features": features, "mask": mask, "labels": labels}


class Trainer:
    """Training loop with early stopping and checkpointing."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = TendencyLoss(
            direction_weight=config.direction_loss_weight,
            confidence_weight=config.confidence_loss_weight,
        )
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self, train_samples: list[dict], val_samples: list[dict]) -> dict:
        """Run full training loop.

        Returns dict with training history.
        """
        train_loader = DataLoader(
            SauronDatasetTorch(train_samples),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            SauronDatasetTorch(val_samples),
            batch_size=self.config.batch_size,
        )

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.max_epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = self._eval_epoch(val_loader)
            history["val_loss"].append(val_loss)

            print(f"Epoch {epoch+1}/{self.config.max_epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n = 0

        for batch in loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(features)

            # Compute loss over sectors with valid labels
            loss = torch.tensor(0.0, device=self.device)
            count = 0
            sector_names = list(SECTORS.keys())

            for i, sector in enumerate(sector_names):
                if sector not in output:
                    continue
                valid = ~labels[:, i].isnan()
                if valid.sum() == 0:
                    continue

                pred_tendency = output[sector]["tendency"][valid]
                pred_confidence = output[sector]["confidence"][valid]
                target = labels[:, i][valid]

                loss = loss + self.loss_fn(pred_tendency, pred_confidence, target)
                count += 1

            if count > 0:
                loss = loss / count
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                output = self.model(features)

                loss = torch.tensor(0.0, device=self.device)
                count = 0
                sector_names = list(SECTORS.keys())

                for i, sector in enumerate(sector_names):
                    if sector not in output:
                        continue
                    valid = ~labels[:, i].isnan()
                    if valid.sum() == 0:
                        continue

                    pred_tendency = output[sector]["tendency"][valid]
                    pred_confidence = output[sector]["confidence"][valid]
                    target = labels[:, i][valid]

                    loss = loss + self.loss_fn(pred_tendency, pred_confidence, target)
                    count += 1

                if count > 0:
                    loss = loss / count

                total_loss += loss.item()
                n += 1

        return total_loss / max(n, 1)

    def _save_checkpoint(self, name: str) -> None:
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / f"{name}.pt")
