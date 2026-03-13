# Sauron: Geo-Economic World Model

## The Eye That Sees All Tendencies

A model that ingests global economic, political, technological, and resource signals
and outputs directional predictions (momentum/tendency scores) across major sectors.

---

## 1. Prediction Targets (What We're Modeling)

We do NOT predict exact prices or GDP numbers. We predict **sector tendency vectors**:
directional momentum scores over configurable time horizons (1mo, 3mo, 6mo, 12mo).

### Sector Tokens (output space)

| Token | What it captures |
|-------|-----------------|
| `NATRES` | Natural resources (oil, gas, mining, metals, timber) |
| `GREEN` | Green energy (solar, wind, hydrogen, EVs, grid storage) |
| `CHIPS` | Semiconductors (fabs, equipment, design, packaging) |
| `SOFTWARE` | Software & cloud (SaaS, infrastructure, AI services) |
| `QUANTUM` | Quantum computing (hardware, algorithms, cryptography) |
| `WEAPONS` | Defense & aerospace (conventional, cyber, space) |
| `EDUCATION` | Education & human capital (universities, edtech, workforce) |
| `BIOTECH` | Biotechnology & pharma (genomics, drug discovery) |
| `FINANCE` | Financial infrastructure (banking, crypto, payments) |
| `INFRA` | Physical infrastructure (construction, logistics, telecom) |
| `AGRI` | Agriculture & food systems |
| `SPACE` | Space economy (launch, satellites, exploration) |

### Output Format Per Sector

```
{
  "sector": "CHIPS",
  "horizon": "3mo",
  "tendency": 0.72,         // [-1, 1] bearish to bullish
  "confidence": 0.65,       // [0, 1]
  "volatility": 0.40,       // [0, 1] expected turbulence
  "drivers": ["TSMC capex", "US export controls", "AI demand"]
}
```

---

## 2. Labeling Strategy

### The Core Problem
We need ground-truth tendency labels for supervised training. Three approaches, in order of pragmatism:

### Approach A: Retrospective Market Labels (Start Here)
- For each sector, build a **basket of representative ETFs/indices**
  - `CHIPS` -> SOXX, SMH; `GREEN` -> ICLN, TAN, QCLN; `WEAPONS` -> ITA, PPA; etc.
- Compute rolling returns over the target horizon (1mo, 3mo, etc.)
- Normalize to [-1, 1] using z-score against historical distribution
- This gives us **automatic labels** for any historical date
- **Pros**: Fully automated, large dataset, clear ground truth
- **Cons**: Markets != real economy, noisy, backward-looking

### Approach B: Composite Index Labels (Phase 2)
- Build custom composite indices per sector mixing:
  - Market data (30%): ETF returns as above
  - Fundamental data (40%): Production volumes, capex, patents filed, policy changes
  - Sentiment data (30%): News volume, tone, expert surveys
- More representative but requires manual calibration per sector

### Approach C: Expert-in-the-Loop (Phase 3)
- Use LLM-assisted labeling: feed Claude/GPT historical snapshots, ask for sector tendency ratings
- Human experts validate and correct
- Expensive but highest quality for ambiguous periods

### Practical Start
**Phase 1 uses Approach A exclusively.** It's fully automatable and gives us thousands of training examples immediately (daily labels going back 10+ years for most sector ETFs).

---

## 3. Data Sources

### Tier 1: Free, API-Accessible, Start Immediately

| Source | Data Type | API? | Update Freq | Notes |
|--------|----------|------|-------------|-------|
| **FRED** (Federal Reserve) | Macro: GDP, inflation, rates, employment, money supply | Yes (free key) | Daily-Monthly | ~800K time series. Best macro source. |
| **World Bank Open Data** | Development indicators, GDP, trade, poverty | Yes (open) | Annual | 16K+ indicators, 200+ countries |
| **Yahoo Finance** (yfinance) | ETF/stock prices, sector indices | Python lib | Real-time | For label generation. Use `yfinance`. |
| **GDELT** | Global news events, sentiment, themes | BigQuery (free) | 15-min | 250M+ events. Geo-coded. Massive. |
| **UN Comtrade** | International trade flows by commodity | Yes (free tier) | Monthly | HS-code level bilateral trade |
| **EIA** (Energy Info Admin) | Oil, gas, coal, renewables production/consumption | Yes (free key) | Weekly-Monthly | US-focused but comprehensive |
| **SIPRI** | Military expenditure, arms transfers | Downloadable CSVs | Annual | Gold standard for defense spending |
| **USPTO/EPO** | Patent filings by technology class | Bulk download | Weekly | Innovation signals per sector |
| **GitHub Archive** | Open source activity by topic | BigQuery (free) | Hourly | Proxy for software/tech trends |
| **OECD Data** | Economic indicators, education, trade | API (free) | Monthly-Annual | 38 member countries + partners |

### Tier 2: Free but Complex Integration

| Source | Data Type | Complexity | Notes |
|--------|----------|-----------|-------|
| **IMF Data** | Balance of payments, fiscal, monetary | Multiple APIs, inconsistent formats | Combine IFS, WEO, DOTS databases |
| **Eurostat** | EU economic/social statistics | SDMX API, learning curve | Very granular EU data |
| **IRENA** | Renewable energy capacity/costs | PDF reports + some data portals | Manual extraction needed |
| **arXiv** | Research paper trends | Bulk download + NLP | Signal for quantum, biotech, AI |
| **SEC EDGAR** | Company filings, earnings | XBRL parsing required | Capex and R&D signals |
| **OpenAlex** | Academic publications & citations | API (free) | Better than Google Scholar API |

### Tier 3: Paid / Premium (Phase 2+)

| Source | Data Type | Cost | Value Add |
|--------|----------|------|-----------|
| **Bloomberg Terminal** | Everything financial | ~$24K/yr | Gold standard, real-time |
| **Refinitiv/Reuters** | News, fundamentals, ESG | ~$15K/yr | Alternative to Bloomberg |
| **S&P Global Market Intelligence** | Sector analytics, supply chains | ~$10K+/yr | Deep sector intelligence |
| **Orbital Insight / Planet Labs** | Satellite imagery analytics | Custom pricing | Physical activity signals |
| **Preqin / PitchBook** | Private market / VC flows | ~$15K+/yr | Early tech investment signals |
| **Quandl (Nasdaq)** | Alternative data bundles | Varies | Aggregated alternative data |

### Recommended Phase 1 Data Stack
```
FRED + yfinance + GDELT + World Bank + EIA + SIPRI CSVs
```
This alone gives us: macro fundamentals, sector price labels, news sentiment, development indicators, energy data, and defense spending. Enough to build and validate the first model.

---

## 4. Framework Decision

### The Contenders

#### PyTorch (Recommended for Phase 1)
- **Pros**: Massive ecosystem, every architecture available, best debugging, HuggingFace integration, community support for time-series (pytorch-forecasting, GluonTS via PyTorch, Chronos)
- **Cons**: Python overhead, deployment needs ONNX/TorchScript export
- **Verdict**: Start here. No contest for research/prototyping phase.

#### Burn (Rust) - Watch for Phase 2+
- **Pros**: Pure Rust, backend-agnostic (WGPU, CUDA, CPU), `no_std` support, growing fast
- **Cons**: Young ecosystem (0.x), limited pre-built architectures, small community, writing custom layers is verbose, no equivalent of HuggingFace datasets/tokenizers
- **Verdict**: Interesting for production inference server. NOT for research phase. Missing too many batteries.

#### Candle (HuggingFace, Rust)
- **Pros**: HuggingFace backing, good for inference of existing models, Rust safety
- **Cons**: Primarily designed for **inference** of existing LLMs, not training custom architectures. Limited training loop support. Optimizers are basic.
- **Verdict**: Wrong tool for this job. Candle is for serving LLMs, not building custom time-series models.

#### JAX/Flax
- **Pros**: Excellent for research, `vmap`/`jit`/`pmap` are powerful, good for custom architectures
- **Cons**: Steeper learning curve, smaller community than PyTorch, Google ecosystem lock-in concerns
- **Verdict**: Valid alternative but PyTorch wins on ecosystem breadth for this use case.

### Decision: PyTorch + Rust Inference Pipeline

```
Phase 1 (Research):  PyTorch  -> train, experiment, validate
Phase 2 (Harden):    PyTorch  -> ONNX export -> Rust inference server (via ort/candle)
Phase 3 (Scale):     Evaluate Burn for full Rust rewrite if model architecture stabilizes
```

The Rust parts of Sauron should be the **data ingestion pipeline** and **inference server**, not the model training. Training benefits too much from PyTorch's flexibility.

---

## 5. Network Architecture

### Starting Architecture: Temporal Fusion Transformer (TFT)

Why TFT:
- Designed specifically for **multi-horizon time-series forecasting**
- Handles **mixed inputs**: static metadata (country, sector) + time-varying known (calendar, scheduled events) + time-varying unknown (prices, sentiment)
- Built-in **interpretability**: variable importance, attention over time
- Proven on similar macro-economic forecasting tasks
- Available in `pytorch-forecasting` library

### Architecture Overview

```
                    ┌──────────────────────────────────────────┐
                    │           SECTOR TENDENCY OUTPUT          │
                    │  [NATRES, GREEN, CHIPS, ..., SPACE] x     │
                    │  [tendency, confidence, volatility]        │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │        Multi-Head Output Layer            │
                    │   (one head per sector, shared backbone)  │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │     Temporal Fusion Transformer Core      │
                    │                                           │
                    │  ┌─────────────────────────────────────┐  │
                    │  │  Multi-Head Self-Attention (time)   │  │
                    │  └─────────────────────────────────────┘  │
                    │  ┌─────────────────────────────────────┐  │
                    │  │  Gated Residual Networks (per var)  │  │
                    │  └─────────────────────────────────────┘  │
                    │  ┌─────────────────────────────────────┐  │
                    │  │  Variable Selection Networks        │  │
                    │  └─────────────────────────────────────┘  │
                    │  ┌─────────────────────────────────────┐  │
                    │  │  LSTM Encoder / Decoder             │  │
                    │  └─────────────────────────────────────┘  │
                    └──────────────┬───────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
 ┌────────▼────────┐   ┌──────────▼──────────┐   ┌────────▼────────┐
 │  Static Inputs   │   │  Known Time-Varying  │   │ Unknown Observed │
 │                  │   │                      │   │                  │
 │ • Country codes  │   │ • Day of week/month  │   │ • FRED series    │
 │ • Sector metadata│   │ • Scheduled releases │   │ • ETF prices     │
 │ • Region groups  │   │ • Election dates     │   │ • GDELT sentiment│
 │                  │   │ • Policy change dates│   │ • Trade volumes  │
 └──────────────────┘   └──────────────────────┘   │ • Energy data    │
                                                    │ • Patent counts  │
                                                    └──────────────────┘
```

### Why Not Just a Plain Transformer?
- Time-series data has **different variable types** (static, known future, unknown) that need special handling
- TFT's variable selection automatically learns which inputs matter for which sectors
- Built-in quantile outputs give us confidence/uncertainty for free
- We get interpretability without sacrificing performance

### Evolution Path
```
v0.1  Simple LSTM baseline (sanity check, 1 week)
v0.2  TFT single-sector (prove the pipeline works, 2 weeks)
v0.3  TFT multi-sector with shared backbone (core model, 4 weeks)
v0.4  Add cross-sector attention (sectors influence each other, 2 weeks)
v0.5  Add news/NLP branch with frozen sentence-transformer (2 weeks)
v1.0  Production model with Rust inference server
```

---

## 6. Project Structure

```
sauron/
├── PLAN.md                          # This file
├── pyproject.toml                   # Python project config
├── sauron/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── sources/                 # One module per data source
│   │   │   ├── fred.py              # FRED API client
│   │   │   ├── yfinance_labels.py   # ETF label generation
│   │   │   ├── gdelt.py             # GDELT news/events
│   │   │   ├── worldbank.py         # World Bank indicators
│   │   │   ├── eia.py               # Energy data
│   │   │   └── sipri.py             # Military spending
│   │   ├── pipeline.py              # Merge, align, normalize all sources
│   │   └── labels.py                # Label generation from ETF baskets
│   ├── model/
│   │   ├── __init__.py
│   │   ├── tft.py                   # Temporal Fusion Transformer
│   │   ├── baseline_lstm.py         # Simple LSTM baseline
│   │   ├── heads.py                 # Per-sector output heads
│   │   └── losses.py                # Custom loss functions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   ├── config.py                # Hyperparameters
│   │   └── evaluate.py              # Evaluation metrics
│   └── inference/
│       ├── __init__.py
│       └── predict.py               # Run predictions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_label_analysis.ipynb
│   └── 03_model_experiments.ipynb
├── data/
│   ├── raw/                         # Downloaded raw data (gitignored)
│   ├── processed/                   # Cleaned, aligned datasets
│   └── sector_etf_baskets.yaml      # ETF basket definitions
├── configs/
│   └── default.yaml                 # Training config
└── tests/
    ├── test_data_sources.py
    └── test_model.py
```

---

## 7. Phased Roadmap

### Phase 1: Data Foundation + Baseline (Weeks 1-4)

**Goal**: Ingest data, generate labels, train a baseline that beats random.

- [ ] Set up project structure, pyproject.toml, dependencies
- [ ] Build FRED data source (macro indicators)
- [ ] Build yfinance label generator (ETF basket returns -> tendency scores)
- [ ] Build GDELT connector (news event counts & sentiment by sector)
- [ ] Build data pipeline: merge sources, align timestamps, handle missing data
- [ ] Train LSTM baseline on 3-5 sectors
- [ ] Evaluate: does it beat a momentum-only baseline?

### Phase 2: Core Model (Weeks 5-8)

**Goal**: TFT model outperforms LSTM on multi-sector prediction.

- [ ] Implement TFT architecture (or adapt from pytorch-forecasting)
- [ ] Multi-sector output heads with shared backbone
- [ ] Add World Bank, EIA, SIPRI data sources
- [ ] Hyperparameter tuning
- [ ] Interpretability analysis: which variables drive which sectors?

### Phase 3: NLP Integration (Weeks 9-12)

**Goal**: News and text data improve predictions.

- [ ] GDELT full-text pipeline with sentence-transformer embeddings
- [ ] Add arXiv abstract embeddings (for quantum, biotech, AI sectors)
- [ ] SEC filing sentiment (for finance, chips, software sectors)
- [ ] Cross-sector attention mechanism

### Phase 4: Production (Weeks 13-16)

**Goal**: Live prediction pipeline.

- [ ] ONNX export of trained model
- [ ] Rust inference server (using `ort` crate for ONNX runtime)
- [ ] Rust data ingestion pipeline (scheduled fetchers)
- [ ] Dashboard / API for querying predictions
- [ ] Monitoring and retraining pipeline

---

## 8. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training framework | PyTorch | Ecosystem, flexibility, community |
| Inference runtime | ONNX via Rust `ort` | Performance + Rust safety |
| Architecture | TFT -> custom multi-sector TFT | Designed for this exact problem |
| Labels | ETF basket z-scored returns | Automatable, immediate, large N |
| Time resolution | Daily (align all sources to daily) | Balances granularity vs. data availability |
| Lookback window | 90 days (tunable) | Enough context without memory issues |
| Prediction horizons | 1mo, 3mo, 6mo | Short enough to validate, long enough to be useful |
| Loss function | Pinball loss (quantile regression) | Gives uncertainty estimates natively |
| Missing data | Forward-fill + masking | Common in multi-source time-series |

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Data quality / gaps | High | Start with most reliable sources (FRED, yfinance). Forward-fill + mask missing. |
| Label noise (markets != economy) | Medium | Phase 2 composite labels. Phase 3 expert validation. |
| Overfitting to historical regimes | High | Walk-forward validation. Regime-aware train/test splits. |
| GDELT volume overwhelms pipeline | Medium | Aggregate to daily sector-level features before training. |
| Too many features, not enough signal | Medium | TFT's variable selection helps. Start with fewer, curated features. |
| Sector definitions are fuzzy | Low | Document ETF baskets clearly. Allow overlap. Validate with domain experts. |

---

## 10. Success Criteria

### Phase 1 Success
- Model produces sector tendency predictions for at least 5 sectors
- Beats momentum baseline (last month's return predicts next month) by >10% on directional accuracy
- Data pipeline ingests from at least 3 sources reliably

### Phase 2 Success
- All 12 sectors covered
- TFT outperforms LSTM baseline by >15% on tendency direction
- Interpretability: can explain top-3 drivers per sector prediction

### Final Success
- Directional accuracy >65% across sectors at 3-month horizon
- Predictions update daily with <5 min latency
- Dashboard shows live tendency vectors with confidence bands
