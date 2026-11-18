# Testbed scorer list (frozen before any testbed Zone C score)

Applies to the trace-driven testbed. Written after the development study on the simulated stream
(`results/dev/eval_zoneC.json`, calibration amendment). Nothing here may change after the first testbed Zone C score.
Testbed Zone C is scored **once**.

## Primary scorer
**BridgeGuard = isolation-forest rank** (`detector.Detector`, `use_lstm: false`, `n_estimators` 300, rank against the
last 15% of Zone A complete hours), conformal p-value against testbed Zone B, alarm at α = 0.05. Seed 0 is primary, seeds 1–2
are the spread. The development table did **not** promote another scorer.

## Comparison scorers (same features, same Zone A fit hours, same Zone B calibration, hyperparameters as in `configs/protocol.yaml` as frozen here)
IF (raw, default) · OCSVM (ν 0.05, RBF) · LOF (k 20) · kNN (k 5) · ECOD · rate threshold · CUSUM (alert rate) ·
KitNET (max cluster 5, 100 epochs) · LSTM-AE (3-h sequences, 16 units). Code: `bridgeguard/baselines.py`.

## Decisions fixed with this list
- **Features:** the testbed's features are chosen by the same label-free rule on testbed Zone A. No hand edits.
- **Gate G2 (amended):** A5-S and A5-R campaign recall ≥ 24/30 on copy 0, plus the one-sided binomial calibration test
  of the clean-Zone C false-alarm count (H0: rate ≤ 0.05, level 0.05). The TPR comparison with the baselines is reported, not gated.
- **Following hours:** on attacked streams, a complete hour that starts within 40 min after an episode ends (the lookback of
  `freq_rolling_std_5w`) is reported separately. It's not an outside-episode false alarm and not a detection. The clean false-alarm
  test uses the uninjected Zone C only, as before.
- **Unscored hours** (LSTM-AE without 2 previous contiguous hours) score −∞ and never alarm. The scored share is reported.

## Testbed generator and zones (decided after the numerics measurement, before any testbed signal value was read)
- **Records:** MIMIC-III Matched Subset numerics with a **1 Hz** header rate only (`testbed/measurements_50patients.json`: HR new every 2 s, SpO₂ 6.5 s, respiration 4 s; minute-rate records change once per minute and have no sub-minute timing). Minute-rate records are left out of these zones. Patients with both rates keep only their 1 Hz records; the two clocks are never stitched.
- **Publishing:** one MQTT message per present timestamp, carrying whichever of HR, SpO₂ (`SpO2` or `%SpO2`) and respiration are present. Nothing is sent when all three are missing, so monitor holds (repeated values) and gaps stay distinct. A threshold crossing adds an emergency message; it never replaces the sample.
- **Zones:** patient-disjoint, from headers only (`testbed/assign_zones.py` → `lockbox/testbed_zones.json`): of the 201 one-second patients in seed-2026 draw order, the first 120 are Zone A, the next 40 Zone B, the last 41 Zone C. This supersedes the cumulative-hours cut of commit 510110c (114/44/43), made before any Zone B/C signal value was read.
- **Ward:** 4 beds per bridge, concurrent on one timeline (an emulation choice). Each bed plays its patients' records back to back, ≥ 1 h apart. Each zone keeps only the prefix in which all 4 beds are occupied; the single-bed tail is a different stream and is dropped. All beds occupied: Zone B 498 h, Zone C 592 h (8 beds would give 215 h and 277 h).
- **What was read before the zones were cut:** the timing measurement read timing and missingness statistics only (no thresholds, no labels) from the 21 patients with 1 Hz records among the first 50 in the draw. All 21 are in Zone A; Zones B and C were untouched.
- **Zone sizes (header span):** A 120 patients / 219 records / 8,756 h; B 40 / 72 / 2,818 h; C 41 / 71 / 2,348 h. Complete-hour counts follow after generation.

## Testbed attacks (frozen before Zones B/C were generated; numbers from Zone A only, `results/testbed/zone_A_bed_rates.json`)
One bed per episode, drawn with the injection seed. Injected alerts use the testbed JSON on `ward/bedN/emergency/<signal>`. **Added** = on top of the bed's own alerts. **Replaced** = the bed's own emergency messages inside the episode are removed (its vitals stream is untouched) and the attack's alerts are published instead.

| Attack | Mechanism | Overlapped ward hour, median (p95) | Rate over the episode |
|---|---|---|---|
| A2 | Added. Development-stream flood timing, alert-anchored 2 h cut | 13,346 | ≈ 19,300/h |
| A5-S | Added. 20 alerts, waits 300–600 s | 32 (56) | 8.4/h |
| A5-R | Added. Development-stream A5 timing only, alert-anchored 2 h cut | 32 (61) | ≈ 9.8/h |
| A5-P | Added. Poisson at 6.77/h, the Zone A per-bed mean | 30 (56) | 6.77/h |
| A5-M | Replaced. Alert-anchored 2 h cut of one Zone A bed's alerts | 29 (52) | ≈ 11.7/h |
| A5-U | Replaced. Poisson at 2.0/h | 23 (45) | 2.0/h |

- A fixed 2 h episode overlaps three clock hours, two of them edges, so the overlapped-hour medians sit below the episode rate.
- A clean Zone A hour is above 50 alerts 4.8% of the time. The added A5 attacks are above 50 on 8.5–13% of overlapped hours. The count is a weak cue, not an absent one. A5-U moves the ward down only because the bed's own alerts are removed: median 23, with 59% of its hours below the ward median of 26.
- A5-P adds a full extra bed's worth of alerts; it isn't matched to the ward rate.
- **Alert-anchored replay** (A2, A5-R, A5-M): the 2 h window starts at a randomly chosen alert, so quiet stretches are undersampled. That's why an A5-M replacement leaves the ward mean at 30.3, not 27.1.
- **Forged alerts** (A5-S, A5-R, A5-P, A5-U) copy the signal, direction and value of a uniformly drawn Zone A emergency. That gives the Zone A mix: respiration 76.2%, SpO₂ 17.3%, heart rate 6.5%. **A5-M** keeps each segment alert's signal and value and only changes the bed id. In both cases the value isn't the bed's vital sample at that second.
- The current development injector writes `retirement/emergency/{heart,fall,inactivity}`. It must be adapted to this payload rule before any testbed injection.
- **Gate G2 unchanged:** campaign recall ≥ 24/30 on copy 0 for A5-S and A5-R, plus the amended calibration test.

## Testbed Zone C result (one look, `results/testbed/eval_zoneC.json`)
Seed 0, copy 0, α = 0.05, 592 clean hours. **G2 fails:** A5-S 2/30, A5-R 9/30 (gate 24/30). The calibration test doesn't reject: 19/592 false alarms, rate 3.2% [1.9%, 5.0%], tail 0.987.
- **BridgeGuard on A5-S, A5-P, A5-M and A5-U is at chance.** An episode overlaps about 3 hours, so false alarms alone would "detect" 1 − (1 − 0.032)³ ≈ 2.8–3.0 of 30 episodes. Hour-level TPR on those attacks (0.02–0.04) is at or below the clean false-alarm rate (0.032). Only A5-R (alert-anchored bursts) is above chance: 9/30, TPR 0.12.
- **No scorer passes.** LOF (A5-S 11/30, A5-R 15/30) and OCSVM (8/30, 13/30) are above their own chance levels (5.4–5.8 and 4.7–5.1) on A5-S and A5-R, at false-alarm rates of 6.4% and 5.6%. The rest are near chance. CUSUM has 0 false alarms and 0 A5 detections. Every scorer detects A2 in 30/30 episodes.
- **Seeds:** BridgeGuard A5-S 2–4/30 and A5-R 7–12/30 across seeds and copies; false alarms 19, 21, 24.
- **Following hours** (A2 copy 0): 23 hours, 2 alarms. Outside-episode false alarms: 16/479.
- **Reading:** a single bed's added or replaced alerts (2–12/h) are hidden in a 4-bed ward hour (mean 32/h on Zones B and C) under hourly whole-bridge features. The development result (A5-S 30/30 against a 2.47/h background) doesn't transfer. This was expected from the Zone A rate analysis and is reported as found.
- **Against each scorer's own chance level** (seed 0, copy 0): the rate rule (7 on A5-S, 8 on A5-R; chance about 3.0 and 2.8), ECOD (7 and 9; same chance) and KitNET (13 on A5-R; chance 5.0) are above chance and still far from 24. **BridgeGuard doesn't beat the rate rule on A5-S** (2 vs 7).
- **CUSUM:** its 0 false alarms are on the clean file. On the A2 copy it alarms on 474 of 479 hours outside the episodes, because the statistic stays high after a flood.
- **LOF's** 38 false alarms are two below the rejection line of 40.
- **Smallest p the forest rank actually returns:** 0.006 (seeds 0 and 1) and 0.008 (seed 2). The training report's 0.002 is the 1/(n + 1) formula. α = 0.01 can fire; the operating point stays α = 0.05.
- **Status:** this is the pre-registered finding, and this look is closed. A per-bed detector would be a new experiment: designed only on Zones A and B, tested once on patients outside A, B and C. It doesn't reopen this score.
