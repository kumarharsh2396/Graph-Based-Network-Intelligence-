# Network Operations Strategy Memo

**Decision requested:** prioritize a targeted hub-and-corridor improvement program and use graph-corrected ETAs for operational planning.

## Executive recommendation

The graph-enhanced ETA reduced holdout MAE from 102.6 to 62.4 minutes (39.2% improvement) and raised trips predicted within 15% of actual from 24.5% to 47.6%. This supports a controlled production pilot, with cold-start routes monitored separately.

Concentrate the first improvement wave on the three highest-ranked hubs below. The ranking combines network dependency, traffic exposure, observed excess minutes, and lack of alternate connectivity; it is not a centrality-only list.

## Five priority hubs

1. **Gurgaon_Bilaspur_HB (Haryana) (IND000000ACB)** — bottleneck score 94.2; approximately 3.96% of observed proxy breaches and 12.14% of network excess movement time are attributed to movements touching this hub.
2. **Bangalore_Nelmngla_H (Karnataka) (IND562132AAA)** — bottleneck score 75.1; approximately 2.91% of observed proxy breaches and 6.15% of network excess movement time are attributed to movements touching this hub.
3. **Bhiwandi_Mankoli_HB (Maharashtra) (IND421302AAG)** — bottleneck score 71.8; approximately 2.69% of observed proxy breaches and 5.29% of network excess movement time are attributed to movements touching this hub.
4. **Pune_Tathawde_H (Maharashtra) (IND411033AAA)** — bottleneck score 59.8; approximately 1.58% of observed proxy breaches and 2.90% of network excess movement time are attributed to movements touching this hub.
5. **Hyderabad_Shamshbd_H (Telangana) (IND501359AAE)** — bottleneck score 59.8; approximately 1.54% of observed proxy breaches and 2.93% of network excess movement time are attributed to movements touching this hub.

## Corridor actions

- **IND110064AAA → IND000000ACB**: median movement is 4.43× OSRM across 26 legs. Assess parallel route and capacity relief.
- **IND847226AAA → IND842001AAA**: median movement is 9.01× OSRM across 14 legs. Audit departure window and facility handoff.
- **IND845305AAA → IND842001AAA**: median movement is 6.03× OSRM across 14 legs. Audit departure window and facility handoff.
- **IND175015AAA → IND174001AAA**: median movement is 7.26× OSRM across 13 legs. Audit departure window and facility handoff.
- **IND847404AAB → IND842001AAA**: median movement is 8.48× OSRM across 13 legs. Audit departure window and facility handoff.

## Expected impact of the first three hub upgrades

Under the explicit scenario that each selected hub removes 30% of its attributable corridor excess and proportionally reduces its attributable breach risk, the historical test window projects 189.5 prevented proxy breaches, a 2.72% reduction, and 52,796 movement minutes recovered.

At the configured value of ₹1,000 per prevented breach, the scenario value is ₹189,525. This is a planning scenario—not booked revenue—until Finance supplies actual SLA penalties and shipment economics.

## FTL versus Carting policy

Use FTL when its predicted time saving, multiplied by the value of a minute saved, exceeds the incremental FTL premium. Recommendations outside corridors where both modes were historically observed should be treated as pilots rather than automatic switches.

The strongest supported profile in this sample is **IND733140AAA → IND733102AAA (night)**, with median modeled FTL savings of 2.4 minutes and a break-even premium of ₹24.

## Controls before rollout

- Pilot ETA correction on high-volume, previously seen corridors and report unseen-node performance separately.
- Validate the top hub recommendations with scan-event and capacity data before approving capital expenditure.
- Replace the 20%-over-OSRM breach proxy with contractual promised-delivery timestamps when available.
- Replace scenario revenue and route premiums with Finance and Procurement values.
- Run controlled FTL/Carting trials; the historical dataset has limited same-corridor mode overlap and is not sufficient for broad causal claims.