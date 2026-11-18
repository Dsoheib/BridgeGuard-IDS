# Per-bed study: protocol (second study; nothing fetched)

**Status.** The ward-level testbed result (`TESTBED_SCORERS.md`, scored once) is closed and stays in the
paper as the pre-registered finding. This is a separate study. It changes the scoring unit from the ward hour to the
bed hour, because the Zone A rate table (`results/testbed/zone_A_bed_rates.json`) showed one bed's alerts disappearing in
a ward hour; the closed look confirmed it. **Design inputs are testbed Zones A and B only. Zone C isn't used to choose
the features, threshold, gate or scorer.** The new patients (Zone D) are scored once. If that look fails, it's the second
result, reported as such.

## 1. Patients (Zone D)
- **Source:** the MIMIC-III Matched Subset numerics, the same seed-2026 shuffle of `RECORDS-numerics` patients
  (`testbed/fetch_numerics.py`). The 400 already drawn are excluded (Zones A, B, C and the 199 without a 1 Hz record).
- **Rule:** walk the shuffle from position 401 on. Keep every patient with at least one 1 Hz numerics record, in order.
  Stop at the first patient after which the 4-bed schedule's all-beds-occupied prefix reaches **≥ 600 h**. This is computed
  from **headers only** (record lengths), before any `.dat` is fetched. No patient is skipped or swapped after its header
  or data is seen. Minute-rate records are dropped, as in A–C.
- **Generation:** the same generator, 4 beds, the same NEWS2 bands and 60 s re-arm, the same all-beds-occupied cut
  (`configs/protocol.yaml`, `testbed`).

## 2. Scoring unit and alarm rule
- **Bed stream:** every message on `ward/bed<b>/…` (vitals and emergencies) of one bed. The windows are the same 60 min / 10 min grid
  from the epoch as the ward, and the features use the same `extract_features`, computed per bed (streamed).
- **Complete bed-hour:** every 6th grid window lying inside one stretch of that bed's stream (no gap > 1 h). The 1 h gaps
  between a bed's records split its stream, so a bed-hour never spans two records.
- **Bed p-value:** conformal against the **pooled Zone B bed-hours** of all 4 beds (beds are exchangeable: patients are
  assigned to beds by the schedule, not by any property).
- **Alarm unit = bed-hour; one false-alarm budget for the ward.** A bed-hour alarms if p ≤ **α / 4 = 0.0125**. A ward hour
  alarms if any of its complete bed-hours alarms. By Bonferroni, the ward-hour false-alarm rate is ≤ α = 0.05 under
  exchangeability. Four separate tests at 0.05 aren't used.
- **Following hours:** the 40-min rule from `TESTBED_SCORERS.md`, applied to the attacked bed's hours.

## 3. Features and scorers (fixed on Zones A and B)
- **Features:** the frozen label-free rule (`configs/protocol.yaml`, `select`, rule revision 1 included) on the Zone A **bed-hour**
  windows of all 4 beds pooled. No hand edits. The result is written before any Zone D file exists.
- **Scorers:** the frozen list (`TESTBED_SCORERS.md`): BridgeGuard = isolation-forest rank (primary), IF raw, OCSVM, LOF,
  kNN, ECOD, rate threshold, CUSUM, KitNET (max cluster 5), LSTM-AE (3-h bed sequences). Same hyperparameters.
- **Fit / calibration:** fit on Zone A bed-hours before the Zone A hold-out time (the last 15% of Zone A's hours, all
  beds); rank reference = the held-out Zone A bed-hours; calibration = all Zone B bed-hours. Seeds 0, 1, 2; seed 0 primary.
- **CUSUM** runs over each bed's consecutive bed-hours and resets at bed-stream gaps.

## 4. Attacks (unchanged from `TESTBED_SCORERS.md`, "Testbed attacks")
- The same six mechanisms, rates, added/replaced modes and payload rule. Forged alerts draw from the **Zone A** pool.
  30 episodes per copy, 3 copies, injection seed **2027** (new zone).
- **One change in placement, forced by the bed unit:** the bed is drawn first, then the episode is placed inside one
  stretch of **that bed's** stream (≥ 1 h from that copy's other episodes). Otherwise an episode could fall in a bed's
  1 h record gap and have no complete bed-hour.

## 5. Evaluation and gate (one look on Zone D)
- **Clean false alarms:** ward hours of uninjected Zone D with ≥ 1 complete bed-hour. Ward-hour alarm rule from §2.
  Calibration test: one-sided exact binomial, H0: ward rate ≤ 0.05, level 0.05. Bed-hour false-alarm rate is also
  reported (expected ≤ 0.0125).
- **Detection:** an episode is detected if **the attacked bed** alarms (p ≤ 0.0125) on a complete bed-hour that overlaps
  the episode. Alarms on other beds don't count as detection. **Denominator: 30 episodes per copy**, never pooled across copies.
- **Gate G3:** seed 0, copy 0: A5-S **and** A5-R campaign recall **≥ 24/30**, and the calibration test doesn't reject.
  A5-P, A5-M, A5-U and A2 are reported, not gated.
- **Chance level** for every cell: 30 × [1 − (1 − bed-hour FPR)^(attacked bed-hours per episode)], reported next to recall.
- **Reported:** both seeds/copies spreads; hour-level TPR on attacked bed-hours; the ward-level table stays unchanged.

## 6. Order of work
1. Commit this protocol. 2. Fetch Zone D **headers** and apply §1 (headers only). 3. Bed-window Zones A and B,
   select features, fit, calibrate; freeze the design. 4. Fetch Zone D data, generate, inject, bed-window. 5. Score once.

## 7. Amendments (committed before any Zone D header is fetched)
Both draft choices stay. The ≥ 600 h stop is a fixed sample-size match to Zone C's 592 occupied hours. Requiring the attacked bed itself to alarm makes G3 a **localization gate**; the denominator stays 30, and an alarm on another bed isn't a detection.

1. **Zone D list in its own file.** The walk starts at **index 400** of the seed-2026 shuffle (10,269 patients; indices 0–399 are the 400 already drawn, including all 201 zone patients). It uses the same schedule and cut as `generator.schedule()`: 1 Hz records only, the 1 h gap after each record, and the all-beds-occupied cut. The list goes to `lockbox/testbed_zone_D.json` (headers in `testbed_data/mimic3_numerics/zoneD_records.json`). `drawn_records.json` and `lockbox/testbed_zones.json` aren't touched. A record the generator can't read stops the run. If the generated cut differs from the header cut, the patient list stays the header list and the difference is reported.
2. **Feature selection** runs on every kept 10-min window of each bed, the 4 beds pooled and **sorted by window start**, so the five stability blocks stay time blocks. Complete bed-hours are the scoring rows. **Hold-out = the ward-hour cut:** bed-hours with hour index < 1861 are fit; bed-hours from hour 1861 on (329 ward hours) are the rank reference.
3. **Placement:** one `place()` call per copy with one `taken` list, so the ≥ 1 h gap holds across beds. Each episode must fit inside one stretch of its drawn bed. Payload timing stays the frozen source-side rule, including the alert-anchored cuts for A5-R, A5-M and A2.
4. **Smallest attainable p:** at the freeze step, record the smallest p the Zone B bed-hour ranks can actually return. If it's above 0.0125, Zone D is scored under the same rule anyway.
5. **Calibration pooling** of the 4 beds stays. Zone A's slot means differ (6.15 / 6.80 / 6.22 / 7.93 alerts/h), so the paper reports the false-alarm rate of **each bed slot**. A ward test that doesn't reject leaves the 5% bound undecided.
6. **Chance level** beside each cell uses the clean-file bed-hour false-alarm rate and the attacked bed's own overlapping bed-hours.
7. **LSTM-AE** sequences stop at a bed-stream gap; a bed-hour without a full sequence scores −∞ and never alarms. Early stopping stays on the held-out Zone A sequences. **CUSUM** keeps the clock hour index, so its gap reset still fires.
8. **A5-U** replaces about 6.8 alerts/h with 2.0. These scorers alarm on the high side, and G3 doesn't gate A5-U. The six frozen rates stay as they are.

## 8. Design result (Zones A and B only; frozen before any Zone D data is fetched)
- **Bed windows:** Zone A 12,838–12,976 windows and 2,017–2,068 complete bed-hours per bed (8,187 in all); Zone B 2,882–2,945 windows and 440–470 bed-hours per bed (1,836).
- **Features** (`results/perbed/selected_features.json`, frozen rule on all 51,662 Zone A bed windows, pooled and sorted by window start): 13 kept. These are alert-rate acceleration, consecutive-emergency count, regularity coefficient, interval variance, burst score, mean inter-alert interval, temporal clustering, topic diversity, rolling std, alert frequency, n_messages, normal/emergency ratio and payload entropy. `n_emergency` is a duplicate of alert frequency. `consecutive_emergency_count` passes here because 15% of bed windows have no alert.
- **Fit / reference / calibration:** 6,954 bed-hours with hour_index < 1861 / 1,233 bed-hours from 1861 on / 1,836 Zone B bed-hours (`results/perbed/train_report.json`, models in `testbed_data/models_perbed/`).
- **Smallest p the Zone B bed-hour ranks can return:** 0.00163 (seeds 0, 1) and 0.00109 (seed 2). That's below 0.0125, so a bed alarm is attainable.
- **Bed-aware scorers** (`bridgeguard/perbed.py`): CUSUM and LSTM-AE run within each bed. The LSTM-AE smoke test on A/B: 6,614 training and 1,170 held-out sequences, all 300 epochs run (no early stop), 94.3% of Zone B bed-hours scorable.

## 9. Zone D result (one look, `results/perbed/eval_zoneD.json`)
Seed 0, copy 0. A bed alarms at p ≤ 0.0125, a ward hour alarms if any bed does, and detection counts only on the attacked bed. **G3 fails:** A5-S **0/30**, A5-R **10/30** (gate 24/30). The ward calibration test doesn't reject: 22/639 ward hours (3.4%, interval 2.2–5.2%, tail 0.976; it rejects at ≥ 42).
- **A5-S is not detected at all:** 0/30 on all 30 covered episodes, 0 of 99 attacked bed-hours (chance 0.9 episodes). Across seeds and copies it's 0–3/30. A forged 8.4 alerts/h on one bed sits inside that bed's own hourly variation (Zone A per bed: mean 6.8, sd 6.3, p95 19).
- **A5-R is detected well above chance but below the gate:** 10/30 (chance 0.8), TPR 0.125. Seeds and copies give 6–11. kNN 17/30, KitNET 18/30, OCSVM 14/30, ECOD 11/30 and the rate rule 7/30 are also far above their chance levels (0.1–1.5). LOF is 1/30.
- **A5-P, A5-M and A5-U are at or near chance for every scorer** (0–3/30). A2: 30/30 for every scorer except the LSTM-AE (28/30).
- **Per-slot bed-hour false-alarm rate** (BridgeGuard seed 0): slot 0 0.33%, slot 1 1.04%, slot 2 0.49%, slot 3 1.79%; pooled 0.91%. Slot 3 is above the 1.25% per-bed target. The ward rate passes the test, but the 5% bound is undecided (section 7.5).
- **Calibration test:** no scorer is rejected. KitNET is closest (41/639, tail 0.064; it rejects at 42). Smallest p returned: 0.00163.
- **Reading:** moving from the ward hour to the bed hour makes the bursty replay (A5-R) visible: 10/30 against 9/30 at ward level, with chance falling from 2.8 to 0.8, and up to 18/30 for KitNET. It doesn't make the steady low-rate forgery (A5-S, A5-P) or the replacement attacks (A5-M, A5-U) visible. This is the second pre-registered result.
- **Corrections.**
  - On the raw event stream, copy 0 of A5-U removes **452** and injects 112; A5-M removes **393** and injects 649. The −326 and +258 are complete-bed-hour totals; the gap is alerts outside complete bed-hours.
  - CUSUM's 0 alarms are on the clean file. On the A2 copy it detects 30/30, then alarms on 731 bed-hours outside the episodes and on all 13 following hours.
  - BridgeGuard's 22 bed-hour alarms fall in 22 distinct ward hours.
  - Slot 3's interval (0.9–3.2%) still covers the 1.25% target.
  - KitNET's 3/30 on A5-P and A5-M is slightly above its chance level of 1.4; the other A5-P, A5-M and A5-U cells are at the null.
- **Status:** this look is closed. No further patient block will be drawn. Both closed results go in the paper.
