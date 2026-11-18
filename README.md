# BridgeGuard: a pre-registered evaluation of label-free alert-injection detection at healthcare MQTT bridges

This repository holds the code, configuration, injection records and protocol documents of the study *A pre-registered
evaluation of label-free alert-injection detection at healthcare MQTT bridges* (Cherih, Lyazid, Kazar).

The question is whether label-free behavioral detectors at an MQTT bridge can see one bedside device injecting false
emergency alerts, or replacing real ones, at a rate that looks ordinary. Ten scorers are evaluated under one protocol:
BridgeGuard (fixed-grid windows, label-free feature selection, an isolation-forest rank and a conformal p-value) and
nine baselines (raw isolation forest, OCSVM, LOF, kNN, ECOD, a calibrated rate threshold, CUSUM, KitNET and an LSTM
sequence autoencoder).

| Study | Test zone | Clean false alarms (BridgeGuard) | Steady | Replay | Gate (24/30 on both) |
|---|---|---|---|---|---|
| Ward level | Zone C | 19 / 592 hours | 2 / 30 | 9 / 30 | fails |
| Bed level | Zone D | 22 / 639 ward hours | 0 / 30 | 10 / 30 | fails |

Every scorer detects at least 28 of 30 floods. No scorer's false-alarm rate is rejected by the exact binomial test.
Against about 27.1 alerts per ward hour (6.8 per bed-hour), a few forged alerts per hour do not stand out in timing or
volume.

## Testbed

The testbed replays the 1 Hz numerics of the MIMIC-III Waveform Database Matched Subset (PhysioNet, Open Database
License; no credentials needed) through four concurrent beds. An emergency is raised when a sample enters a NEWS2
score-3 band. Patients are split by seeded order, from record headers only, into disjoint zones:

| Zone | Patients | Use |
|---|---|---|
| A | 120 | feature selection, fitting, rank reference |
| B | 40 | conformal calibration |
| C | 41 | ward-level test (hashed and locked before scoring) |
| D | 33 | bed-level test (drawn after the Zone C look, under its own protocol) |

The data are regenerated from PhysioNet by the scripts and are not versioned. The hashes of every generated zone and
injection file are in `lockbox/`.

## Protocol

Each protocol document fixes its rules before the data they govern are read:

| Document | What it fixes |
|---|---|
| `protocol/PROTOCOL.md` | Score, complete-hour grid, alpha = 0.05, gates, and the calibration amendment with the original verdict kept |
| `protocol/TESTBED_SCORERS.md` | Testbed scorer list, generator rules and the six attack mechanisms, before Zones B and C existed |
| `protocol/PERBED_PROTOCOL.md` | Bed-level unit, Zone D rule and design, fixed on Zones A and B only |
| `configs/protocol.yaml` | Every parameter |

## Names in the paper and in the files

| Paper | Result files and protocol documents |
|---|---|
| Flood | A2 |
| Steady | A5-S |
| Replay | A5-R |
| Poisson | A5-P |
| Mimic | A5-M |
| Sparse | A5-U |
| BridgeGuard | `BridgeGuard (IF rank)` |

## Layout

| Path | Contents |
|---|---|
| `bridgeguard/` | Label firewall, zones and lock box, windowing and features, label-free selection, detector, baselines, injection, evaluation |
| `testbed/` | MIMIC-III fetch, zone assignment, generator, windowing, ward-level and bed-level runners |
| `run_dev.py` | Development run on the low-background simulated stream |
| `configs/protocol.yaml` | Every parameter of the protocol |
| `lockbox/` | Zone lists, data manifests and hashes, injection records |
| `results/` | `testbed/` (ward level), `perbed/` (bed level), `dev/` (development run) |
| `protocol/` | Protocol documents |
| `paper/` | `make_tables.py` and `make_figures.py`, which write every table, number and figure of the paper |
| `tests/` | Label-leakage, zone, generator, injection, baseline and bed-level tests |

## Reproducing

Python 3.13; dependencies are in `requirements.txt`. Run from the repository root.

```
python -m pytest -q tests
python paper/make_tables.py       # paper/tables/*.tex and paper/numbers.tex from results/, lockbox/ and configs/
python paper/make_figures.py      # paper/figures/*.pdf (the rate figure also reads the generated Zone A events)
```

The step-by-step commands for the testbed, both studies and the development run are in
[`REPRODUCE.md`](REPRODUCE.md).

## License

MIT; see `LICENSE`. The MIMIC-III Waveform Database Matched Subset is distributed by PhysioNet under the Open Database
License.
