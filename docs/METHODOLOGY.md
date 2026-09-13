# Methodology and Reproducibility Notes

## Unit of analysis

The raw CSV contains cumulative checkpoint rows. A modeling observation is one
unique `(split, trip, source, destination, OD start)` leg. Cumulative actual,
OSRM, distance, and scan-to-scan fields use the maximum recorded value. Segment
sums are intentionally not used because the supplied data contains corrections,
including negative segment times.

The primary ETA target is `actual_elapsed_min` (`start_scan_to_end_scan`). The
diagnostic decomposition is:

```text
movement excess = max(actual movement - OSRM, 0)
dwell/handling proxy = max(scan-to-scan elapsed - actual movement, 0)
```

## Graph definition

- Directed node: facility ID
- Directed edge: observed source-to-destination facility movement
- Edge risk: median actual-movement/OSRM-time ratio
- Chronic corridor: median ratio above 1.20 with at least five observations
- Edge strata: corridor, route type, and departure time bucket

Unweighted and OSRM-time-weighted betweenness, in/out-degree, directed
clustering, weak-component size, volume, breach exposure, and excess minutes are
computed. The bottleneck rank assigns half its weight to realized breach/excess
impact so that a lightly used but central node does not dominate a proven delay
hotspot.

## ETA benchmark

The supplied `training` and `test` split is chronological and is never
reshuffled. The training period is split again by trip creation time for model
selection; every trip remains entirely on one side of this validation split.

Models:

1. OSRM time without correction.
2. Ridge baseline on departure-known OSRM, route-type, hour, and weekday fields.
3. Two-layer mean-aggregation GraphSAGE edge regressor using the same fields,
   training-only smoothed corridor/node history, and source/destination graph
   representations.

Historical features for training rows are leave-one-out. Validation/test
features use only the preceding fit window. Graph topology, centralities,
normalization, and endpoint mappings are also built from training data only.
Unseen corridors and unseen nodes are reported as separate test segments.

The optimization target is standardized `log1p(actual_elapsed_min)` with a
Smooth-L1 loss. Final metrics are calculated in minutes after reversing the
transformation. Primary metrics are MAE and percentage of predictions within
15% of actual.

## Route-mode framework

Each profile is scored twice by changing only the route-type field. The FTL
break-even premium is:

```text
max(predicted Carting ETA - predicted FTL ETA, 0) × value per minute
```

The recommendation is supported only if both route types occurred historically
on the same corridor. One-mode and unseen corridors are labeled low-confidence
scenarios requiring a controlled operational pilot. The supplied observational
data does not identify a causal treatment effect.

## Upgrade simulation

The default scenario selects the top three training-period hubs and assumes each
removes 30% of its attributable corridor excess and breach risk. Corridor excess
and breach responsibility are split equally across endpoints to prevent double
counting. Financial value is a scenario using a configurable value per prevented
breach, not a revenue forecast.

## Known limitations

- The data covers roughly three weeks in 2018, so annual seasonality and current
  network conditions are unobserved.
- No promised ETA or contractual SLA field is present; 20% over OSRM is a proxy.
- No package counts, capacity, cost, revenue, or SLA-penalty fields are present.
- Only a small number of corridors have both FTL and Carting observations.
- Facility coordinates are absent, so the SVG is a topology map rather than a
  geographic map.

