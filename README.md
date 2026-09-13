# Graph-Based Network Intelligence for Delivery ETA Prediction

> Improving delivery ETA accuracy by modelling the logistics network as a directed graph and combining graph analytics with machine learning, operational simulation, and business recommendations.

---

## Live Dashboard

**[Open the deployed dashboard](PASTE_YOUR_DEPLOYED_STREAMLIT_LINK_HERE)**

Replace the link above with the URL provided by Streamlit Community Cloud after deployment.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Project Objectives](#project-objectives)
4. [Dataset](#dataset)
5. [Project Workflow](#project-workflow)
6. [Methodology](#methodology)
7. [Model Results](#model-results)
8. [FTL versus Carting Framework](#ftl-versus-carting-framework)
9. [Hub Upgrade Simulation](#hub-upgrade-simulation)
10. [Generated Artifacts](#generated-artifacts)
11. [Repository Structure](#repository-structure)
12. [Installation](#installation)
13. [Run the Analytical Pipeline](#run-the-analytical-pipeline)
14. [Run the Streamlit Dashboard](#run-the-streamlit-dashboard)
15. [Deploy to Streamlit Community Cloud](#deploy-to-streamlit-community-cloud)
16. [Limitations](#limitations)
17. [How Another Company Can Use This Methodology](#how-another-company-can-use-this-methodology)
18. [Key Business Impact](#key-business-impact)

---

## Project Overview

This project improves delivery ETA prediction using graph-based network intelligence.

The logistics network is represented as a directed graph:

- Facilities are graph nodes.
- Delivery corridors are directed edges.
- Historical travel performance is stored as edge-level information.
- Network structure is used to identify bottleneck hubs and delayed corridors.

The project combines machine learning, graph analytics, operational simulation, and business recommendations into a single reproducible pipeline with an interactive dashboard.

---

## Business Problem

Traditional routing engines such as OSRM estimate travel time using distance and road conditions. Actual delivery time is also affected by:

- Facility dwell time
- Loading and unloading delays
- Congestion
- Route type
- Departure time
- Hub capacity
- Network bottlenecks
- Lack of alternate routes

When ETA predictions are inaccurate:

- SLAs are missed.
- Customer satisfaction decreases.
- Capacity planning becomes unreliable.
- Operations teams cannot identify the main sources of delay.

The objective is to build a model that improves ETA accuracy and identifies the hubs and corridors causing network-wide delays.

---

## Project Objectives

1. Construct a directed logistics graph.
2. Aggregate raw checkpoint data into delivery-leg records.
3. Identify high-risk hubs and corridors.
4. Compare OSRM, a traditional baseline model, and GraphSAGE.
5. Build an FTL-versus-Carting decision framework.
6. Simulate the impact of upgrading high-risk hubs.
7. Produce an operations strategy memo and dashboard.

---

## Dataset

The dataset is stored at:

```text
data/delivery_data.csv
```

### Dataset Statistics

| Metric | Value |
|---|---|
| Checkpoint records | 144,867 |
| Trips | 14,817 |
| Delivery legs | ~26,369 |
| Facilities | 1,657 |
| Directed corridors | 2,783 |

### Dataset Characteristics

- FTL and Carting route types
- OSRM and actual travel-time fields
- Departure timestamps and route information
- Repeated cumulative checkpoint rows
- Multi-stop trips and facility transfer events

Because the raw records contain repeated cumulative checkpoints, the project first converts them into one canonical record per facility-to-facility leg.

---

## Project Workflow

```text
Raw Checkpoints
      ↓
Canonical Delivery Legs
      ↓
Directed Logistics Graph
      ↓
Graph Feature Engineering
      ↓
ETA Prediction Models
      ↓
Bottleneck Identification
      ↓
Route Optimization Analysis
      ↓
Operational Simulation
      ↓
Business Recommendations
```

---

## Methodology

### 1. Data Preparation

Repeated checkpoint rows are grouped using:

```text
data
trip_uuid
source_center
destination_center
od_start_time
```

Cumulative fields use their maximum value to represent the completed leg.

**Primary target**

```text
actual_elapsed_min = start_scan_to_end_scan
```

**Diagnostic fields**

```text
actual_movement_min
osrm_time_min
dwell_time_min
movement_delay_ratio
movement_excess_min
```

**Dwell-time proxy**

```text
max(actual_elapsed_min - actual_movement_min, 0)
```

---

### 2. Graph Construction

The graph is directed:

```text
Source facility → Destination facility
```

Each corridor stores:

- Observation count
- Median OSRM time
- Median actual movement time
- Median delay ratio
- Breach rate
- Excess movement time
- Route-type coverage
- Time-of-day performance

**Chronically delayed corridor definition**

A corridor is considered chronically delayed when:

```text
median actual movement time / median OSRM time > 1.20
```

and it has sufficient historical observations.

---

### 3. Bottleneck Analysis

**Graph metrics**

- In-degree
- Out-degree
- Betweenness centrality
- Weighted betweenness centrality
- Clustering coefficient
- Weak connected-component size

**Operational metrics**

- Traffic exposure
- Delay exposure
- Breach contribution

The final bottleneck score combines structural importance with observed operational impact. This prevents a highly connected but fast facility from outranking a smaller facility that causes significant delay.

---

### 4. ETA Models

Three approaches are compared.

#### OSRM Benchmark

Uses the original OSRM time estimate directly.

#### Tabular Baseline

Ridge regression using departure-known features:

- OSRM time
- OSRM distance
- OSRM speed
- Route type
- Departure hour
- Weekday
- Weekend indicator

#### GraphSAGE Model

Uses:

- Baseline trip features
- Historical corridor delay features
- Source and destination facility history
- Facility degree
- Betweenness centrality
- Clustering
- Network component size
- GraphSAGE node representations

**Leakage controls**

- Graph features are computed from training data only.
- Training records use leave-one-out historical features to avoid target leakage.

---

## Model Results

Chronological test-set performance:

| Model | MAE (minutes) | Within 15% of Actual |
|---|---|---|
| OSRM | 207.5 | 0.4% |
| Tabular baseline | 102.6 | 24.5% |
| GraphSAGE | **62.4** | **47.6%** |

**Key findings**

- The GraphSAGE model reduced MAE by approximately **39%** compared with the tabular baseline.
- Graph features capture operational behaviour not visible in traditional route features.
- Unseen corridors and unseen facilities are evaluated separately because they are harder to predict.

---

## FTL versus Carting Framework

The model estimates ETA under both route types and produces:

- Predicted Carting ETA
- Predicted FTL ETA
- FTL time saving
- Break-even FTL premium
- Recommended route type

**Time saving**

```text
FTL saving = Predicted Carting ETA - Predicted FTL ETA
```

**Break-even premium**

```text
FTL time saving × value per minute saved
```

**Evidence levels**

| Level | Meaning |
|---|---|
| Supported | Both modes observed on the corridor |
| Low confidence | Only one mode observed |
| Unsupported | Unseen corridor |

Because route type was not randomly assigned, the recommendations are predictive scenarios rather than guaranteed causal effects.

---

## Hub Upgrade Simulation

The project simulates improvements to the top three bottleneck hubs.

**Default assumptions**

- 30% reduction in attributable delay risk
- 20%-over-OSRM rule used as a breach proxy
- ₹1,000 value per prevented breach

**Simulation outputs**

- Projected late-delivery reduction
- Movement minutes saved
- Expected prevented breaches
- Scenario revenue recovered

These are planning estimates. Actual financial impact requires real revenue, SLA penalty, capacity, and upgrade-cost data.

---

## Generated Artifacts

The analytical pipeline automatically generates:

```text
canonical_legs.csv
bottleneck_hubs.csv
delayed_corridors.csv
model_metrics.csv
test_predictions.csv
route_mode_recommendations.csv
hub_upgrade_summary.json
network_bottlenecks.svg
network_operations_strategy_memo.md
```

---

## Repository Structure

```text
.
├── app.py                    # Streamlit dashboard entry point (or aap.py)
├── config.yaml               # Pipeline parameters, thresholds, financial assumptions
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── data/
│   └── delivery_data.csv     # Raw checkpoint-level delivery records
│
├── artifacts/                             # Auto-generated pipeline outputs
│   ├── canonical_legs.csv                 # Deduplicated facility-to-facility legs
│   ├── bottleneck_hubs.csv                # Ranked facilities by bottleneck score
│   ├── delayed_corridors.csv              # Corridors with delay ratio > 1.20
│   ├── model_metrics.csv                  # OSRM vs Baseline vs GraphSAGE performance
│   ├── test_predictions.csv               # Chronological holdout predictions
│   ├── route_mode_recommendations.csv     # FTL vs Carting decisions and premiums
│   ├── hub_upgrade_summary.json           # Intervention simulation results
│   ├── network_bottlenecks.svg            # Network topology visualization
│   └── network_operations_strategy_memo.md# Executive strategy memo
│
├── src/
│   └── gbni/
│       ├── data_pipeline.py   # Checkpoint aggregation into canonical legs
│       ├── graph_analysis.py  # Directed graph construction and centrality metrics
│       ├── features.py        # Feature engineering with leave-one-out history
│       ├── models.py          # Ridge baseline and GraphSAGE model definitions
│       ├── benchmark.py       # Model evaluation and comparison
│       ├── bottlenecks.py     # Composite bottleneck scoring
│       ├── route_decision.py  # FTL vs Carting decision framework
│       ├── simulation.py      # Hub upgrade impact scenarios
│       ├── reporting.py       # Metrics export and memo generation
│       └── visualization.py   # Network and performance plots
│
└── tests/                     # Unit tests for pipeline and model modules
```

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Run the Analytical Pipeline

From the project root:

```bash
python -m gbni.cli run \
  --data data/delivery_data.csv \
  --output artifacts \
  --epochs 80
```

This generates the cleaned data, graph summaries, model predictions, bottleneck rankings, route recommendations, intervention scenarios, SVG visualization, and strategy memo. All outputs are written to the `artifacts/` directory.

---

## Run the Streamlit Dashboard

If the file is named `aap.py`:

```bash
streamlit run aap.py
```

If renamed to `app.py`:

```bash
streamlit run app.py
```

The dashboard reads the generated files from the `artifacts/` directory and includes:

- ETA model KPIs
- Model comparison charts
- Network bottleneck visualization
- Hub ranking
- Delayed corridor table
- FTL-versus-Carting recommendations
- Hub upgrade scenario metrics
- Strategy memo
- Filtered prediction download

---

## Deploy to Streamlit Community Cloud

1. Upload the project to GitHub.
2. Ensure the repository includes:
   - `app.py` or `aap.py`
   - `requirements.txt`
   - `artifacts/`
   - `data/`
3. Open [share.streamlit.io](https://share.streamlit.io).
4. Select the GitHub repository.
5. Select the branch.
6. Set the main file to `app.py` or `aap.py`.
7. Click **Deploy**.
8. Copy the generated URL into the **Live Dashboard** section at the top of this README.

---

## Limitations

- The dataset covers only a short historical period.
- Contractual SLA fields are not available.
- Revenue, shipment value, route costs, and upgrade costs are not available.
- FTL and Carting overlap is limited on individual corridors.
- The model is trained on historical relationships and must be retrained for another company.
- The network visualization is a topology map, not a geographic map.

---

## How Another Company Can Use This Methodology

The framework is reusable by substituting domain-specific components:

| Component in this project | Replace with |
|---|---|
| Facilities | Its own entities |
| Corridors | Its own relationships |
| Actual delivery time | Its own target variable |
| OSRM estimate | Its own baseline estimate |
| Revenue assumptions | Its own financial values |

The overall process remains:

```text
Raw events
   → canonical records
   → directed graph
   → graph features
   → baseline model
   → graph model
   → bottleneck analysis
   → intervention simulation
   → business dashboard
```

The code structure is reusable, but the model must be retrained using the company's own data.

---

## Key Business Impact

This framework enables logistics teams to:

- Improve ETA accuracy
- Prioritize infrastructure investments
- Detect bottleneck facilities
- Identify chronically delayed corridors
- Optimize transportation mode selection
- Quantify operational improvement opportunities
- Improve SLA compliance
- Enhance customer experience

By combining machine learning with graph intelligence, organizations gain a network-wide understanding of delivery performance rather than examining individual routes in isolation.
