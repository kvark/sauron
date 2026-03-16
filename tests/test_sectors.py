"""Tests for sector definitions."""

from sauron.sectors import SECTORS, SectorDef


def test_all_sectors_defined():
    expected = {
        "NATRES", "GREEN", "CHIPS", "SOFTWARE", "QUANTUM", "WEAPONS",
        "EDUCATION", "BIOTECH", "FINANCE", "INFRA", "AGRI", "SPACE",
    }
    assert set(SECTORS.keys()) == expected


def test_sector_has_etfs():
    for token, sector in SECTORS.items():
        assert len(sector.etf_basket) > 0, f"{token} has no ETF basket"


def test_sector_has_keywords():
    for token, sector in SECTORS.items():
        assert len(sector.keywords) > 0, f"{token} has no keywords"


def test_sector_def_is_frozen():
    sector = SECTORS["CHIPS"]
    assert isinstance(sector, SectorDef)
