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

### Design Philosophy: Foundation Model + Domain Layers

Recent time-series foundation models (Chronos-2, MOIRAI-2) already solve the core
forecasting problem — "given this history, what comes next?" — with strong zero-shot
performance. **We don't need to reinvent that.** Our unique value is the domain-aware
layers that these models lack entirely:

1. **Event-driven regime detection** (geopolitical shocks → forecast adjustments)
2. **Cross-sector causal propagation** (energy crisis → trade disruption → financial stress)
3. **Scenario simulation** ("what if X sanctions Y?" counterfactual engine)
4. **Policy-level interpretability** (why did the forecast change, in geopolitical terms)

### Foundation Model Selection

| Model | Params | Strengths | Role in Sauron |
|-------|--------|-----------|---------------|
| **Chronos-2** (Amazon) | 120M | Native covariate support via group attention, strong long-horizon, 300+ forecasts/sec on A10G, 600M+ HF downloads | **Primary backbone** — feeds sector time series as groups with covariates |
| **MOIRAI-2** (Salesforce) | 11M | Decoder-only, 30x smaller than predecessor, #1 MASE on GIFT-Eval, fast inference | **Lightweight baseline** — fast iteration, ensemble candidate |
| **TFT** (custom) | ~5-10M | Built-in variable selection, interpretability, handles mixed input types | **Domain-specific head** — sits on top of foundation embeddings |

**Neither Chronos-2 nor MOIRAI-2 handles:**
- Domain-specific causal structure (sanctions → trade disruption propagation)
- Discrete geopolitical event encoding (regime shifts from GDELT)
- Cross-sector interaction modeling (energy ↔ trade ↔ finance feedback loops)
- Counterfactual scenario simulation ("what if China invades Taiwan?")
- Policy-level explainability ("GDP forecast dropped because energy imports fell 40%")

**That's our model's entire unique contribution.**

### Architecture Overview: Hybrid Stack

```
                    ┌──────────────────────────────────────────┐
                    │           SECTOR TENDENCY OUTPUT          │
                    │  [NATRES, GREEN, CHIPS, ..., SPACE] x     │
                    │  [tendency, confidence, volatility]        │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │       LAYER 5: Interpretability           │
                    │  Driver attribution per sector prediction │
                    │  ("CHIPS ↓ because US export controls +   │
                    │   TSMC capex delay + AI demand plateau")  │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │    LAYER 4: Scenario Simulation Engine    │
                    │  Counterfactual injection: perturb event  │
                    │  inputs and re-propagate through layers   │
                    │  "What if EU sanctions on X? Re-run L2-3" │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │  LAYER 3: Cross-Sector Interaction Graph  │
                    │  Sector-to-sector attention + learned      │
                    │  causal adjacency (energy→trade→finance)  │
                    │  Models spillover effects and feedback     │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │   LAYER 2: Event-Driven Regime Encoder    │
                    │  GDELT events → event embeddings →        │
                    │  regime shift detection → forecast         │
                    │  adjustments (sanctions, wars, coups,      │
                    │  trade agreements, elections)              │
                    └──────────────┬───────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────┐
                    │   LAYER 1: Foundation Forecast Backbone    │
                    │                                           │
                    │  ┌────────────────┐  ┌─────────────────┐ │
                    │  │   Chronos-2    │  │    MOIRAI-2     │ │
                    │  │   (primary)    │  │   (ensemble)    │ │
                    │  │  Group attn    │  │   Lightweight   │ │
                    │  │  w/ covariates │  │   baseline      │ │
                    │  └───────┬────────┘  └───────┬─────────┘ │
                    │          └────────┬───────────┘           │
                    │          Ensemble / selection             │
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
 │ • Trade network  │   │ • Policy change dates│   │ • Trade volumes  │
 │   topology       │   │ • Sanctions calendar │   │ • Energy data    │
 └──────────────────┘   └──────────────────────┘   │ • Patent counts  │
                                                    │ • Military spend │
                                                    └──────────────────┘
```

### Layer Details

**Layer 1 — Foundation Forecast Backbone:**
Use Chronos-2's group attention to feed related time series (e.g., GDP + trade + energy
for a country-sector pair) as a group. The model learns cross-series dynamics zero-shot.
Fine-tune on our geo-economic data to adapt temporal patterns to this domain. MOIRAI-2
serves as an independent lightweight forecast for ensembling.

**Layer 2 — Event-Driven Regime Encoder:**
GDELT events are encoded into dense vectors via a learned event embedding layer. A regime
detection module identifies structural breaks (sanctions imposed, wars started, agreements
signed) and generates regime-shift signals that modulate Layer 1's forecasts. This is where
our model diverges fundamentally from general-purpose forecasters — it understands that
"Russia invades Ukraine" is not just a data point but a regime change.

**Layer 3 — Cross-Sector Interaction Graph:**
A graph attention network where nodes are sectors and edges represent learned causal
relationships. Energy → Trade, Military → Finance, Chips → Software, etc. The graph
propagates shocks: an energy sector disruption flows through the graph to affect downstream
sectors with learned lag structures. This is the "sector spillover" model.

**Layer 4 — Scenario Simulation Engine:**
Accepts hypothetical event injections: "What if EU imposes full sanctions on country X?"
The event is encoded via Layer 2, propagated through the sector graph in Layer 3, and
produces counterfactual forecasts. This is the "what if" engine — completely absent from
all existing foundation models.

**Layer 5 — Interpretability & Attribution:**
Integrated gradients + attention weight analysis to produce human-readable driver
explanations. For each sector prediction, output: "CHIPS tendency ↓0.3 because:
US export controls (+0.15), TSMC capex delay (+0.10), AI demand plateau (+0.05)."

### Why This Hybrid Beats Pure Approaches

| Approach | Problem |
|----------|---------|
| Foundation models alone | No domain structure, no events, no scenarios, no interpretability |
| TFT from scratch | Reinvents the forecasting engine; months of work that Chronos-2 does better |
| Pure causal modeling | Too rigid; can't learn latent patterns from data |
| **Our hybrid** | Foundation models handle temporal forecasting; our layers add domain intelligence |

### Evolution Path
```
v0.1  Chronos-2 zero-shot baseline (prove data pipeline works, 1 week)
v0.2  Chronos-2 fine-tuned on our sector data (2 weeks)
v0.3  Add MOIRAI-2 ensemble + event encoder (Layer 2, 3 weeks)
v0.4  Add cross-sector interaction graph (Layer 3, 3 weeks)
v0.5  Add scenario simulation engine (Layer 4, 2 weeks)
v0.6  Add interpretability/attribution (Layer 5, 2 weeks)
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
│   │   ├── backbone.py              # Chronos-2 / MOIRAI-2 foundation model wrappers
│   │   ├── event_encoder.py         # Layer 2: GDELT event embeddings + regime detection
│   │   ├── sector_graph.py          # Layer 3: Cross-sector interaction graph (GAT)
│   │   ├── scenario_engine.py       # Layer 4: Counterfactual scenario simulation
│   │   ├── attribution.py           # Layer 5: Driver attribution + interpretability
│   │   ├── heads.py                 # Per-sector output heads
│   │   ├── ensemble.py              # Foundation model ensemble logic
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

### Phase 1: Data Foundation + Foundation Model Baseline (Weeks 1-4)

**Goal**: Ingest data, generate labels, establish strong baselines using pre-trained models.

- [ ] Set up project structure, pyproject.toml, dependencies
- [ ] Build FRED data source (macro indicators)
- [ ] Build yfinance label generator (ETF basket returns -> tendency scores)
- [ ] Build GDELT connector (news event counts & sentiment by sector)
- [ ] Build data pipeline: merge sources, align timestamps, handle missing data
- [ ] Run Chronos-2 zero-shot on sector time series (baseline 1)
- [ ] Run MOIRAI-2 zero-shot as lightweight baseline (baseline 2)
- [ ] Fine-tune Chronos-2 on our sector data with covariates
- [ ] Evaluate: do foundation models beat momentum baseline? (expect yes)

### Phase 2: Domain Layers (Weeks 5-10)

**Goal**: Build the layers that differentiate us from generic forecasters.

- [ ] **Layer 2 — Event Encoder**: GDELT event embeddings + regime shift detection
- [ ] **Layer 3 — Sector Graph**: Cross-sector GAT with learned causal adjacency
- [ ] Add World Bank, EIA, SIPRI data sources as additional covariates
- [ ] Ensemble logic: Chronos-2 + MOIRAI-2 weighted combination
- [ ] Wire layers together: foundation backbone → events → sector graph → output heads
- [ ] Evaluate: does event-aware model outperform foundation-only baseline?
- [ ] Interpretability analysis: which events and cross-sector links matter most?

### Phase 3: Scenario Engine + NLP (Weeks 11-14)

**Goal**: Counterfactual simulation works. News text improves event detection.

- [ ] **Layer 4 — Scenario Engine**: counterfactual event injection + propagation
- [ ] **Layer 5 — Attribution**: integrated gradients + attention-based explanations
- [ ] GDELT full-text pipeline with sentence-transformer embeddings for richer events
- [ ] arXiv abstract embeddings (quantum, biotech, AI sector signals)
- [ ] SEC filing sentiment (finance, chips, software sector signals)
- [ ] Validate scenarios: do known historical shocks produce correct counterfactuals?

### Phase 4: Production (Weeks 15-18)

**Goal**: Live prediction pipeline with scenario API.

- [ ] ONNX export of domain layers (foundation models already have optimized inference)
- [ ] Rust inference server (using `ort` crate for ONNX runtime)
- [ ] Rust data ingestion pipeline (scheduled fetchers)
- [ ] Scenario API: accept event hypotheticals, return counterfactual forecasts
- [ ] Dashboard showing live tendency vectors + confidence bands + driver attribution
- [ ] Monitoring and retraining pipeline

---

## 8. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Training framework | PyTorch | Ecosystem, flexibility, HuggingFace integration |
| Inference runtime | ONNX via Rust `ort` | Performance + Rust safety |
| Forecasting backbone | Chronos-2 (primary) + MOIRAI-2 (ensemble) | Don't reinvent forecasting; leverage 120M pre-trained params |
| Domain layers | Custom (event encoder, sector graph, scenario engine) | Our unique value — absent from all foundation models |
| Labels | ETF basket z-scored returns | Automatable, immediate, large N |
| Time resolution | Daily (align all sources to daily) | Balances granularity vs. data availability |
| Lookback window | 90 days (tunable) | Enough context without memory issues |
| Prediction horizons | 1mo, 3mo, 6mo | Short enough to validate, long enough to be useful |
| Loss function | Pinball loss (quantile regression) | Gives uncertainty estimates natively |
| Missing data | Forward-fill + masking | Common in multi-source time-series |
| Scenario simulation | Counterfactual event injection | Key differentiator: "what if" analysis |

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
- Chronos-2 zero-shot produces sector tendency predictions for at least 5 sectors
- Fine-tuned Chronos-2 beats momentum baseline by >10% on directional accuracy
- Data pipeline ingests from at least 3 sources reliably

### Phase 2 Success
- All 12 sectors covered
- Event-aware model outperforms foundation-only baseline by >10% on tendency direction
- Cross-sector graph captures known relationships (energy → trade validated)
- Interpretability: can explain top-3 drivers per sector prediction

### Phase 3 Success
- Scenario engine produces plausible counterfactuals for known historical events
  (e.g., 2022 Russia-Ukraine: inject event → model predicts energy/trade disruption)
- NLP-enriched events improve regime detection accuracy
- Attribution layer produces human-readable explanations

### Final Success
- Directional accuracy >65% across sectors at 3-month horizon
- Predictions update daily with <5 min latency
- Dashboard shows live tendency vectors with confidence bands + driver attribution
- Scenario API responds to "what if" queries in <10 seconds
