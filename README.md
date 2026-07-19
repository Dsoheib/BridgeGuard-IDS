# BridgeGuard: Behavioral Detection of Slow-Poisoning Attacks in Secure IoT Messaging

BridgeGuard is a behavioral intrusion detection framework designed to detect **slow poisoning attacks in IoT messaging systems**, even when the attacker uses **valid TLS certificates** and bypasses traditional authentication mechanisms.

Unlike signature-based or volumetric anomaly detectors, BridgeGuard models **temporal behavioral patterns of IoT alert streams** using lightweight machine learning.

The framework combines:

* **Isolation Forest** for behavioral anomaly detection
* **LSTM temporal modeling** for sequential consistency
* **Ensemble scoring with calibration** for stable decision thresholds

BridgeGuard targets **resource-constrained edge deployments** and has been evaluated on multiple IoT datasets.

---

# Key Contributions

BridgeGuard addresses a critical gap in IoT security:

Traditional protections (TLS, authentication, access control) ensure **who can send messages**, but do not ensure **whether the behavior of a device remains legitimate over time**.

BridgeGuard detects attacks where:

* an attacker steals or misuses a valid certificate
* publishes realistic alerts
* slowly manipulates the system behavior

This attack model is referred to as **slow poisoning**.

BridgeGuard introduces:

1. Behavioral feature extraction from MQTT alert streams
2. Attack-aware feature selection
3. Hybrid anomaly detection (Isolation Forest + LSTM)
4. Ensemble calibration
5. Cross-domain evaluation on external IoT datasets
6. Adversarial robustness evaluation
7. Edge-deployment feasibility analysis

---

# Repository Structure

The repository is organized into modular components corresponding to the experimental pipeline.

```
BridgeGuard
│
├── data_pipeline
│   └── window_extraction.py
│
├── feature_engineering
│   └── attack_aware_feature_selection.py
│
├── models
│   ├── iforest_hyperparameter_tuning.py
│   ├── train_iforest.py
│   └── train_lstm.py
│
├── evaluation
│   ├── evaluate_bridgeguard.py
│   └── statistical_validation.py
│
├── interpretability
│   └── interpretability_analysis.py
│
├── baselines
│   └── baseline_ocsvm_crossdomain.py
│
├── cross_domain
│   └── cross_domain_evaluation.py
│
├── attack_scenarios
│   └── attack5_slow_poisoning.py
│
├── edge_benchmark
│   └── benchmark_latency.py
│
├── bridgeguard_features
├── bridgeguard_models
├── figures
├── stats_results
└── shap_lime_results
```

---

# Experimental Pipeline

The BridgeGuard pipeline is executed in the following order.

## 1. Window Extraction

```
data_pipeline/window_extraction.py
```

This script converts raw MQTT alert logs into **temporal windows** suitable for machine learning.

Each window contains behavioral statistics extracted from alert streams, such as:

* alert frequency
* inter-alert intervals
* temporal clustering patterns
* alert rate acceleration
* periodicity indicators

Output:

```
bridgeguard_features/features_selected_labeled.csv
```

---

## 2. Attack-Aware Feature Selection

```
feature_engineering/attack_aware_feature_selection.py
```

This step selects the most discriminative behavioral features using:

* statistical filtering
* stability analysis
* cross-validation consistency

Selected features are stored in:

```
bridgeguard_features/selected_features.json
```

---

## 3. Isolation Forest Hyperparameter Optimization

```
models/iforest_hyperparameter_tuning.py
```

Isolation Forest parameters are tuned using a custom objective based on:

* lower confidence bound stability
* cross-validation robustness

This produces the optimal hyperparameters used for final training.

---

## 4. Isolation Forest Training

```
models/train_iforest.py
```

The final Isolation Forest model is trained using the selected features.

Output:

```
bridgeguard_models/iforest_model.pkl
```

The model learns the behavioral distribution of normal alert streams.

---

## 5. LSTM Temporal Model Training

```
models/train_lstm.py
```

The LSTM model captures **temporal dependencies across alert windows**.

Configuration:

* sequence length: 10
* behavioral feature input
* anomaly probability output

Output:

```
bridgeguard_models/lstm_model_selected.keras
```

---

## 6. BridgeGuard Evaluation

```
evaluation/evaluate_bridgeguard.py
```

This script evaluates the combined ensemble on the test set.

Metrics reported:

* AUC
* TPR
* FPR
* F1-score
* confusion matrix

Results are exported to:

```
bridgeguard_models/final_paper_metrics.json
```

> **Note on calibration artifacts.** The shipped
> `bridgeguard_models/lstm_temperature.json`, `platt_lstm.pkl`, and
> `final_paper_metrics.json` are the **paper-era artifacts**
> (T* = 1.0000, gating delta = 0.65, Youden threshold = 0.949,
> TPR = 91.8%, FPR = 0.8%, AUC = 0.9967), matching the values reported
> in the paper. Running `evaluate_bridgeguard.py` **refits** the Platt
> calibrators, the temperature T*, and the gating parameters from
> Zone-B data and overwrites these files; depending on the exact
> library stack, the refit may converge to marginally different values
> (e.g., T* within 3e-4 of unity) with metrics within the paper's
> reported confidence intervals. To restore the paper-era calibration,
> revert these three files from git.

---

## 7. Statistical Validation

```
evaluation/statistical_validation.py
```

Formal statistical validation includes:

* Clopper-Pearson confidence intervals
* McNemar test
* Wilcoxon signed-rank test
* bootstrap confidence intervals

Output:

```
stats_results/formal_stats.json
```

---

# Model Interpretability

```
interpretability/interpretability_analysis.py
```

BridgeGuard provides explainability through:

* SHAP KernelExplainer attributions (Isolation Forest)
* SHAP GradientExplainer attributions (LSTM; replaces LIME, which is
  incompatible with recurrent sequence inputs)

Outputs:

```
shap_lime_results/
```

These results show which behavioral features contribute most strongly to anomaly detection.

---

# Baseline Comparison

```
baselines/baseline_ocsvm_crossdomain.py
```

BridgeGuard is compared against **One-Class SVM** across multiple IoT datasets.

Evaluation includes:

* cross-domain detection performance
* robustness to adversarial noise
* calibration comparison

---

# Cross-Domain Evaluation

```
cross_domain/cross_domain_evaluation.py
```

BridgeGuard is evaluated on external datasets such as:

* IoT-Fridge
* IoT-Thermostat
* IoT-Weather

The results demonstrate that behavioral features generalize across IoT devices.

---

# Attack Scenario Simulation (Optional)

```
attack_scenarios/attack5_slow_poisoning.py
```

This script simulates the **slow poisoning attack described in the paper**.

The attacker:

* uses a valid TLS certificate
* injects plausible alerts
* publishes them slowly to avoid detection

Environment variables allow configuration:

```
BG_BROKER_HOST
BG_BROKER_PORT
BG_CERTS_DIR
BG_FAST_MODE
```

The script generates realistic attack sequences for evaluation.

---

# Edge Deployment Benchmark (Optional)

```
edge_benchmark/benchmark_latency.py
```

This script evaluates BridgeGuard inference latency.

Steps:

1. Convert the LSTM model to TensorFlow Lite
2. Benchmark local inference latency
3. Estimate Raspberry Pi 4 latency via analytical scaling
4. Generate latency figures and reports

Outputs:

```
rpi4_latency_results/
figures/latency_profile.pdf
```

---

# Reproducing the Results

Full pipeline, in execution order:

```
python feature_engineering/attack_aware_feature_selection.py
python data_pipeline/window_extraction.py
python data_pipeline/sequential_features.py
python models/iforest_hyperparameter_tuning.py
python models/train_iforest.py
python models/train_lstm.py
python models/regen_iforest_figure.py
python models/regen_lstm_figure.py
python evaluation/evaluate_bridgeguard.py
python evaluation/statistical_validation.py
python evaluation/block_bootstrap_ci.py
python evaluation/regen_ensemble_figure.py
python evaluation/greybox_jitter_evaluation.py
python evaluation/seed_variance_runner.py
python baselines/baseline_ocsvm_crossdomain.py
python cross_domain/cross_domain_evaluation.py
python cross_domain/STEP_TONIOT_ADAPTATION.py
python cross_domain/ocsvm_crossdomain_comparison.py
python cross_domain/baselines_crossdomain_comparison.py
python cross_domain/mqttset_crossdomain.py
python interpretability/interpretability_analysis.py
python edge_benchmark/benchmark_latency.py
```

`attack_scenarios/attack5_slow_poisoning.py` requires a live MQTT broker in
an isolated testbed and is not part of the offline pipeline.

---

# Requirements

Tested environment (see `requirements.txt`):

```
Python 3.11
tensorflow==2.20.0
scikit-learn==1.8.0
numpy==2.4.2
shap==0.50.0
pandas, matplotlib, seaborn, scipy, tqdm, joblib
paho-mqtt   (only for the optional live attack-scenario script)
```

---

# Citation

If you use BridgeGuard in your research, please cite:

```
@article{bridgeguard2026,
  title   = {BridgeGuard: Behavioral Detection of Sub-Threshold Semantic
             Attacks in Healthcare MQTT Bridges via Calibrated Edge Ensembles},
  author  = {Cherih, Soheib and Sabri, Lyazid},
  journal = {Future Generation Computer Systems},
  note    = {Under review},
  year    = {2026}
}
```

---

# Script-to-Paper Mapping

| Paper artifact | Generating script |
|---|---|
| Feature discriminability figure + feature table | `feature_engineering/attack_aware_feature_selection.py` |
| IForest calibration figure | `models/train_iforest.py` |
| LSTM training figure | `models/train_lstm.py` |
| Performance / ablation tables, ensemble figure | `evaluation/evaluate_bridgeguard.py` |
| Confidence-interval figure | `evaluation/statistical_validation.py` |
| Block-bootstrap FPR CI | `evaluation/block_bootstrap_ci.py` |
| Grey-box jitter table + figure | `evaluation/greybox_jitter_evaluation.py` |
| LSTM seed-stability paragraph | `evaluation/seed_variance_runner.py` |
| OCSVM comparison table | `baselines/baseline_ocsvm_crossdomain.py` |
| Cross-domain tables (ToN-IoT) | `cross_domain/STEP_TONIOT_ADAPTATION.py`, `cross_domain/cross_domain_evaluation.py` |
| SHAP / GradientExplainer panels | `interpretability/interpretability_analysis.py` |
| Edge latency appendix | `edge_benchmark/benchmark_latency.py` |

# Random Seeds and Determinism

Primary seed: `42` (enforced for NumPy/TensorFlow in each script).
Seed-stability study: `42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021`
(pre-trained per-seed LSTM checkpoints in `bridgeguard_models/seed_<seed>/`).

# Data Availability

* `bridgeguard_features/` — extracted behavioral window features used by all
  training and evaluation scripts (included in this repository; no payload
  content, no PHI).
* Raw healthcare MQTT captures — not redistributed; available from the
  corresponding author upon reasonable request (see paper, Code and Data
  Availability).
* ToN-IoT sensor telemetry (`toniot_data/IoT_Fridge.csv`,
  `IoT_Thermostat.csv`, `IoT_Weather.csv`) — download from
  https://research.unsw.edu.au/projects/toniot-datasets and place in
  `toniot_data/`.
* MQTTset (`MQTTset/Data/CSV/`) — download from
  https://www.kaggle.com/datasets/cnrieiit/mqttset and place in `MQTTset/`.

---

# License

MIT License

---

# Contact

For questions or collaboration inquiries, please open an issue in the repository.
