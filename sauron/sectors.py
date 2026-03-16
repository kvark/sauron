"""Sector definitions and ETF basket mappings for label generation."""

from dataclasses import dataclass

SECTORS: dict[str, "SectorDef"] = {}


@dataclass(frozen=True)
class SectorDef:
    token: str
    name: str
    etf_basket: list[str]  # ticker symbols for label generation
    keywords: list[str]  # for GDELT / news classification


def _register(token: str, name: str, etfs: list[str], keywords: list[str]) -> None:
    SECTORS[token] = SectorDef(token=token, name=name, etf_basket=etfs, keywords=keywords)


# --- Sector registry ---

_register(
    "NATRES",
    "Natural Resources",
    ["XLE", "XOP", "GDX", "SLV", "DBA"],
    ["oil", "gas", "mining", "metals", "timber", "commodities", "crude", "petroleum"],
)

_register(
    "GREEN",
    "Green Energy",
    ["ICLN", "TAN", "QCLN", "FAN", "LIT"],
    ["solar", "wind", "hydrogen", "EV", "renewable", "clean energy", "grid storage", "battery"],
)

_register(
    "CHIPS",
    "Semiconductors",
    ["SOXX", "SMH", "PSI"],
    [
        "semiconductor", "chip", "fab", "wafer", "TSMC", "ASML",
        "lithography", "packaging", "EUV",
    ],
)

_register(
    "SOFTWARE",
    "Software & Cloud",
    ["IGV", "WCLD", "SKYY", "CLOU"],
    ["SaaS", "cloud", "software", "AI services", "infrastructure", "data center"],
)

_register(
    "QUANTUM",
    "Quantum Computing",
    ["QTUM"],
    [
        "quantum computing", "qubit", "quantum cryptography", "quantum algorithm",
        "quantum supremacy", "quantum error correction",
    ],
)

_register(
    "WEAPONS",
    "Defense & Aerospace",
    ["ITA", "PPA", "XAR", "DFEN"],
    [
        "defense", "military", "weapons", "aerospace", "missile", "drone",
        "cyber warfare", "arms", "NATO",
    ],
)

_register(
    "EDUCATION",
    "Education & Human Capital",
    ["EDUT"],
    [
        "education", "university", "edtech", "workforce", "student",
        "training", "human capital", "enrollment",
    ],
)

_register(
    "BIOTECH",
    "Biotechnology & Pharma",
    ["XBI", "IBB", "BBH", "ARKG"],
    [
        "biotech", "pharma", "genomics", "drug discovery", "clinical trial",
        "FDA", "gene therapy", "CRISPR",
    ],
)

_register(
    "FINANCE",
    "Financial Infrastructure",
    ["XLF", "KBE", "ARKF", "BITQ"],
    [
        "banking", "fintech", "crypto", "payments", "insurance",
        "blockchain", "stablecoin", "CBDC",
    ],
)

_register(
    "INFRA",
    "Physical Infrastructure",
    ["PAVE", "IFRA", "IGF"],
    [
        "infrastructure", "construction", "logistics", "telecom",
        "bridge", "highway", "5G", "broadband", "port",
    ],
)

_register(
    "AGRI",
    "Agriculture & Food Systems",
    ["MOO", "DBA", "VEGI"],
    [
        "agriculture", "farming", "food", "crop", "fertilizer",
        "irrigation", "food security", "grain",
    ],
)

_register(
    "SPACE",
    "Space Economy",
    ["UFO", "ARKX", "ROKT"],
    [
        "space", "satellite", "launch", "orbit", "SpaceX",
        "rocket", "constellation", "NASA",
    ],
)
