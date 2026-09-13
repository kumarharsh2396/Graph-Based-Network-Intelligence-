# Graph-Based Network Intelligence for Delivery ETAs

This repository builds a reproducible logistics-network intelligence system from
`data/delivery_data.csv`. It converts cumulative checkpoint rows into canonical
facility-to-facility legs, constructs a directed network, audits bottlenecks,
benchmarks ETA models, evaluates FTL-versus-Carting scenarios, and produces
operations-ready outputs.

## Quick start

```powershell
python -m unittest discover -s tests -v
python -m gbni.cli run --data data/delivery_data.csv --output artifacts --epochs 80
```

The package lives under `src/`. For a source checkout, the repository includes a
small root-level `gbni` bootstrap package so the commands above work without an
installation step.

## Modeling rules

- A model record is one OD leg, not one raw checkpoint row.
- The supplied chronological `training` and `test` split is preserved.
- Graph aggregates and embeddings used for prediction are learned from training
  data only.
- The primary ETA target is scan-to-scan elapsed time; movement delay and dwell
  time are retained for diagnosis.
- "SLA breach" is reported as a configurable proxy because contractual SLA and
  promised-delivery fields are not present in the supplied data.
- Financial impact is scenario-based because shipment revenue and route costs
  are not present.

## Output folders

After a run, `artifacts/` contains cleaned leg and graph tables, model metrics,
bottleneck rankings, route-mode scenarios, intervention estimates, a network
SVG, trained model state, and an operations strategy memo.

# Graph-Based-Network-Intelligence-for-Delivery-ETAs
