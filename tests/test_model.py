"""Tests for model components (layers 2-5, heads, losses)."""

import torch

from sauron.model.event_encoder import EventEncoder, RegimeDetector
from sauron.model.heads import MultiSectorHead, SectorHead
from sauron.model.losses import PinballLoss, TendencyLoss
from sauron.model.sector_graph import SectorInteractionGraph


def test_event_encoder_shapes():
    enc = EventEncoder(num_event_types=300, num_countries=250, embedding_dim=128)
    batch, max_events = 4, 10

    out = enc(
        event_types=torch.randint(0, 300, (batch, max_events)),
        goldstein=torch.randn(batch, max_events),
        mentions=torch.randn(batch, max_events),
        tone=torch.randn(batch, max_events),
        actor1_country=torch.randint(0, 250, (batch, max_events)),
        actor2_country=torch.randint(0, 250, (batch, max_events)),
        event_mask=torch.ones(batch, max_events, dtype=torch.bool),
    )
    assert out.shape == (batch, 128)


def test_regime_detector():
    det = RegimeDetector(embedding_dim=128, window_size=7)
    batch, days = 4, 7

    event_seq = torch.randn(batch, days, 128)
    prob, emb = det(event_seq)

    assert prob.shape == (batch, 1)
    assert emb.shape == (batch, 128)
    assert (prob >= 0).all() and (prob <= 1).all()


def test_sector_graph_shapes():
    graph = SectorInteractionGraph(input_dim=128, hidden_dim=128, num_heads=4, num_layers=2)
    batch, num_sectors = 4, 12

    features = torch.randn(batch, num_sectors, 128)
    out = graph(features)
    assert out.shape == (batch, num_sectors, 128)


def test_sector_graph_with_regime():
    graph = SectorInteractionGraph(input_dim=128, hidden_dim=128)
    batch, num_sectors = 4, 12

    features = torch.randn(batch, num_sectors, 128)
    regime_emb = torch.randn(batch, 128)
    out = graph(features, regime_embedding=regime_emb)
    assert out.shape == (batch, num_sectors, 128)


def test_sector_head():
    head = SectorHead(input_dim=128)
    x = torch.randn(4, 128)
    out = head(x)
    assert out["tendency"].shape == (4,)
    assert out["confidence"].shape == (4,)
    assert (out["tendency"] >= -1).all() and (out["tendency"] <= 1).all()
    assert (out["confidence"] >= 0).all() and (out["confidence"] <= 1).all()


def test_multi_sector_head():
    head = MultiSectorHead(input_dim=128)
    batch, num_sectors = 4, 12
    x = torch.randn(batch, num_sectors, 128)
    out = head(x)
    assert "CHIPS" in out
    assert "tendency" in out["CHIPS"]


def test_pinball_loss():
    loss_fn = PinballLoss(quantiles=[0.1, 0.5, 0.9])
    preds = torch.randn(8, 3)
    targets = torch.randn(8)
    loss = loss_fn(preds, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_tendency_loss():
    loss_fn = TendencyLoss()
    pred_tendency = torch.randn(8)
    pred_confidence = torch.sigmoid(torch.randn(8))
    target = torch.randn(8)
    loss = loss_fn(pred_tendency, pred_confidence, target)
    assert loss.dim() == 0
    assert loss.item() >= 0
