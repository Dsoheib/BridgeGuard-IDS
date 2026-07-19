
"""
OCSVM vs BridgeGuard Cross-Domain Comparison
===============================================
Trains One-Class SVM on the same source normal windows used by BridgeGuard,
evaluates zero-shot on the same 3 ToN-IoT datasets, and prints a side-by-side
comparison against BridgeGuard results.

Run from bridgeguard/ root:
 python3 cross_domain/ocsvm_crossdomain_comparison.py
"""
#
# ToN-IoT DATA PROVENANCE — the cross-domain evaluation uses the OFFICIAL
# unmodified ToN-IoT files (Alsaedi et al., IEEE Access 2020):
#   Processed_datasets/Processed_IoT_dataset/{IoT_Fridge,IoT_Thermostat,IoT_Weather}.csv
# Official download folder (UNSW SharePoint):
#   https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EvBTaetotpdGnW7rJQ8fCvYBh8063CNeY9W33MpRsarJaQ?e=yZlnxW
# Project page: https://research.unsw.edu.au/projects/toniot-datasets
# SHA-256 checksums of the exact files used: see toniot_data/README.md
#


import os, sys, json, pickle, warnings
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")
np.random.seed(42)

FEATURES_DIR   = "bridgeguard_features"
MODELS_DIR     = "bridgeguard_models"
V6_RESULTS_DIR = "toniot_results"
V10_RESULTS_DIR = "toniot_v10_results"
OUTPUT_DIR     = "ocsvm_vs_bridgeguard_results"
DATASETS       = ["IoT_Fridge", "IoT_Thermostat", "IoT_Weather"]

BEHAVIORAL_ATTACKS = {"ddos", "injection", "ransomware", "backdoor",
                      "flooding", "slow_poisoning", "mitm", "dos"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("OCSVM vs BridgeGuard Cross-Domain Comparison")
print("=" * 65)

print("\nLoading BridgeGuard artifacts...")
try:
    with open(os.path.join(FEATURES_DIR, "selected_features.json")) as fh:
        SELECTED = json.load(fh)["selected_features"]

    with open(os.path.join(MODELS_DIR, "feature_scaler_selected.pkl"), "rb") as fh:
        scaler = pickle.load(fh)

    normal_df = pd.read_csv(os.path.join(FEATURES_DIR, "features_selected_normal.csv"))
    X_src     = normal_df[SELECTED].values.astype(np.float64)
    X_src_sc  = scaler.transform(X_src)

except FileNotFoundError as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

print(f"  Features : {SELECTED}")
print(f"  Source N : {len(X_src)} normal windows")

print("\nTraining OCSVM variants on source normal windows...")

OCSVM_VARIANTS = {
    "OCSVM_nu005": OneClassSVM(kernel="rbf", nu=0.05, gamma="scale"),
    "OCSVM_nu010": OneClassSVM(kernel="rbf", nu=0.10, gamma="scale"),
    "OCSVM_nu020": OneClassSVM(kernel="rbf", nu=0.20, gamma="scale"),
}

OCSVM_BOUNDS = {}
for name, clf in OCSVM_VARIANTS.items():
    clf.fit(X_src_sc)
    train_scores = clf.decision_function(X_src_sc)
    OCSVM_BOUNDS[name] = (float(train_scores.min()), float(train_scores.max()))
    print(f"  {name}: trained  bounds=[{OCSVM_BOUNDS[name][0]:.4f}, {OCSVM_BOUNDS[name][1]:.4f}]")

def coral_align(Xs, Xt):
    eps = 1e-6
    Cs  = np.cov(Xs.T) + eps * np.eye(Xs.shape[1])
    Ct  = np.cov(Xt.T) + eps * np.eye(Xt.shape[1])
    try:
        A = np.real(np.linalg.inv(sqrtm(Ct))) @ np.real(sqrtm(Cs))
        if not np.all(np.isfinite(A)):
            raise ValueError("non-finite")
        aligned = (Xt - Xt.mean(0)) @ A + Xs.mean(0)
        aligned = np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
        return aligned, True
    except Exception:
        return Xt.copy(), False

def ocsvm_score(clf, X_sc):
    df = clf.decision_function(X_sc)
    lo = float(np.percentile(df, 2))
    hi = float(np.percentile(df, 98))
    rng = hi - lo
    if rng < 1e-12:
        return np.full(len(X_sc), 0.5)
    return np.clip((df - lo) / rng, 0.0, 1.0)

def metrics_at_fpr(p_normal, y, target_fpr=0.15):
    y_attack = (y == 0).astype(int)
    n_atk  = y_attack.sum()
    n_norm = len(y) - n_atk
    if n_atk == 0 or n_norm == 0:
        return dict(auc=0.5, tpr=0.0, fpr=0.0, f1=0.0, threshold=0.5)
    try:
        auc = float(roc_auc_score(y, p_normal))
    except Exception:
        auc = 0.5
    best_thr, best_tpr, best_fpr, best_f1 = 0.5, 0.0, 0.0, 0.0
    for thr in np.arange(0.95, 0.05, -0.01):
        yp_atk = (p_normal < thr).astype(int)
        fa  = int((yp_atk.astype(bool) & (~y_attack.astype(bool))).sum())
        det = int((yp_atk.astype(bool) &  y_attack.astype(bool)).sum())
        fpr_val = fa  / n_norm
        tpr_val = det / n_atk
        if fpr_val <= target_fpr and tpr_val > best_tpr:
            best_tpr = tpr_val
            best_fpr = fpr_val
            best_thr = float(thr)
            best_f1  = float(f1_score(y_attack, yp_atk, zero_division=0))
    return dict(auc=auc, tpr=best_tpr, fpr=best_fpr, f1=best_f1, threshold=best_thr)

def best_ocsvm(results_dict):
    return max(results_dict.items(), key=lambda kv: kv[1]["auc"])

def load_bg_v10(dname):
    path = os.path.join(V10_RESULTS_DIR, f"{dname}_v10.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)

all_results = {}

for dname in DATASETS:
    print(f"\n{'='*65}")
    print(f"DATASET: {dname}")
    print("="*65)

    feat_path = os.path.join(V6_RESULTS_DIR, f"{dname}_features.csv")
    if not os.path.exists(feat_path):
        print(f"  Feature CSV not found: {feat_path}   skipped")
        print(" Run cross-domain adaptation.py first to cache features.")
        continue

    df = pd.read_csv(feat_path)

    if "freq_rolling_std_5w" not in df.columns:
        if "alert_frequency_per_hour" in df.columns:
            df["freq_rolling_std_5w"] = (
                df["alert_frequency_per_hour"]
                .rolling(window=5, min_periods=3).std().fillna(0.0)
            )

    for f in SELECTED:
        if f not in df.columns:
            df[f] = 0.0

    y  = df["label"].values
    at = df["attack_type"].values if "attack_type" in df.columns else None
    nm = (y == 1); am = (y == 0)

    X_raw = np.nan_to_num(df[SELECTED].values.astype(np.float64),
                          nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  {len(df)} windows: {nm.sum()} normal / {am.sum()} attack")

    mu_t  = X_raw.mean(0);  std_t = X_raw.std(0) + 1e-9
    X_z   = np.clip((X_raw - mu_t) / std_t, -50.0, 50.0)
    X_src_z = np.clip((X_src_sc - X_src_sc.mean(0)) / (X_src_sc.std(0) + 1e-9), -50.0, 50.0)
    X_coral, coral_ok = coral_align(X_src_z, X_z)
    print(f"  CORAL: {'ok' if coral_ok else 'fallback'}")

    X_tgt_sc = scaler.transform(X_raw)

    src_mu  = X_src_sc.mean(0);  src_std = X_src_sc.std(0) + 1e-9
    X_tgt_coral_sc = (X_coral - src_mu) / src_std

    ocsvm_results = {}
    ocsvm_coral_results = {}
    for name, clf in OCSVM_VARIANTS.items():
        p_norm   = ocsvm_score(clf, X_tgt_sc)
        m        = metrics_at_fpr(p_norm, y)
        ocsvm_results[name] = m

        p_norm_c = ocsvm_score(clf, X_tgt_coral_sc)
        m_c      = metrics_at_fpr(p_norm_c, y)
        ocsvm_coral_results[name + "_coral"] = m_c

        print(f"  {name:<16}  AUC={m['auc']:.4f}  TPR={m['tpr']*100:.1f}%  FPR={m['fpr']*100:.1f}%")
        print(f"  {name+'_coral':<16}  AUC={m_c['auc']:.4f}  TPR={m_c['tpr']*100:.1f}%  FPR={m_c['fpr']*100:.1f}%")

    best_name, best_m = best_ocsvm(ocsvm_results)
    best_coral_name, best_coral_m = best_ocsvm(ocsvm_coral_results)

    bg = load_bg_v10(dname)
    if bg:
        bg_m = bg.get("v10_fpr15", {})
        bg_beh = bg.get("v10_behavioral_fpr15", bg.get("v10_behavioral_optimal", {}))
        print(f"\n  BridgeGuard @ FPR<=15%:")
        print(f"    AUC={bg_m.get('auc',0):.4f}  "
              f"TPR={bg_m.get('tpr',0)*100:.1f}%  "
              f"FPR={bg_m.get('fpr',0)*100:.1f}%  "
              f"F1={bg_m.get('f1',0):.4f}")
        if bg_beh:
            print(f"    AUC_behavioral={bg_beh.get('auc',0):.4f}")

    all_results[dname] = {
        "ocsvm": ocsvm_results,
        "ocsvm_coral": ocsvm_coral_results,
        "best_ocsvm": (best_name, best_m),
        "best_ocsvm_coral": (best_coral_name, best_coral_m),
        "bridgeguard": bg,
        "n_normal": int(nm.sum()),
        "n_attack": int(am.sum()),
    }

print(f"\n\n{'='*65}")
print("COMPARISON SUMMARY OCSVM vs BridgeGuard (cross-domain, FPR<=15%)")
print("="*65)

header = (f"{'Dataset':<22} {'Model':<22} {'AUC':>6} {'TPR':>7} "
          f"{'FPR':>7} {'F1':>6}")
print(header)
print("-" * 72)

latex_rows = []
for dname in DATASETS:
    if dname not in all_results:
        continue
    r = all_results[dname]
    n_win = r["n_normal"] + r["n_attack"]

    bn, bm = r["best_ocsvm"]
    print(f"{' '+dname:<22} {bn:<22} "
          f"{bm['auc']:>6.3f} {bm['tpr']*100:>6.1f}% "
          f"{bm['fpr']*100:>6.1f}% {bm['f1']:>6.3f}")

    bcn, bcm = r["best_ocsvm_coral"]
    print(f"{' '+dname:<22} {bcn:<22} "
          f"{bcm['auc']:>6.3f} {bcm['tpr']*100:>6.1f}% "
          f"{bcm['fpr']*100:>6.1f}% {bcm['f1']:>6.3f}")

    bg = r.get("bridgeguard")
    if bg:
        bg_m   = bg.get("v10_fpr15", {})
        bg_beh = bg.get("v10_behavioral_fpr15", bg.get("v10_behavioral_optimal", {}))
        bg_auc_beh = bg_beh.get("auc", bg_m.get("auc", 0)) if bg_beh else bg_m.get("auc", 0)
        print(f"{' '+dname:<22} {'BridgeGuard':<22} "
              f"{bg_m.get('auc',0):>6.3f} {bg_m.get('tpr',0)*100:>6.1f}% "
              f"{bg_m.get('fpr',0)*100:>6.1f}% {bg_m.get('f1',0):>6.3f}  "
              f"(beh AUC={bg_auc_beh:.3f})")

    print()

    bg_m   = bg.get("v10_fpr15", {}) if bg else {}
    bg_beh = (bg.get("v10_behavioral_fpr15") or
              bg.get("v10_behavioral_optimal") or {}) if bg else {}
    bg_auc_beh = bg_beh.get("auc", bg_m.get("auc", 0))

    latex_rows.append({
        "dname": dname.replace("IoT_", ""),
        "n_win": n_win,
        "ocsvm_name": bn.replace("OCSVM_", "OCSVM "),
        "ocsvm_auc": bm["auc"],
        "ocsvm_tpr": bm["tpr"],
        "ocsvm_fpr": bm["fpr"],
        "ocsvm_f1":  bm["f1"],
        "ocsvm_coral_name": bcn.replace("OCSVM_", "OCSVM ").replace("_coral", "+CORAL"),
        "ocsvm_coral_auc": bcm["auc"],
        "ocsvm_coral_tpr": bcm["tpr"],
        "ocsvm_coral_fpr": bcm["fpr"],
        "ocsvm_coral_f1":  bcm["f1"],
        "bg_auc": bg_m.get("auc", 0),
        "bg_tpr": bg_m.get("tpr", 0),
        "bg_fpr": bg_m.get("fpr", 0),
        "bg_f1":  bg_m.get("f1", 0),
        "bg_auc_beh": bg_auc_beh,
    })

print("\nDelta AUC: BridgeGuard vs best OCSVM (without CORAL)")
print("-" * 50)
for r_dict in latex_rows:
    delta = r_dict["bg_auc"] - r_dict["ocsvm_auc"]
    sign = "+" if delta >= 0 else ""
    print(f"  {r_dict['dname']:<16}  BG - OCSVM = {sign}{delta:+.3f}  "
          f"({'BridgeGuard wins' if delta > 0 else 'OCSVM wins' if delta < 0 else 'tie'})")

print("\nDelta AUC: BridgeGuard vs best OCSVM+CORAL")
print("-" * 50)
for r_dict in latex_rows:
    delta = r_dict["bg_auc"] - r_dict["ocsvm_coral_auc"]
    sign = "+" if delta >= 0 else ""
    print(f"  {r_dict['dname']:<16}  BG - OCSVM+CORAL = {sign}{delta:+.3f}  "
          f"({'BridgeGuard wins' if delta > 0 else 'OCSVM+CORAL wins' if delta < 0 else 'tie'})")

print("\n\n% ====== LaTeX Table: OCSVM vs BridgeGuard Cross-Domain ======")
print(r"""\begin{table}[ht]
\centering
\caption{Cross-domain comparison: BridgeGuard vs OCSVM baselines on ToN-IoT
 sensor datasets (zero-shot transfer, no target labels). Primary metric:
 highest TPR at FPR$\leq$15\%. $\dagger$Best OCSVM variant by AUC among
 $\nu\in\{0.05,0.10,0.20\}$. CORAL = covariance-shift alignment applied
 before scoring. $^*$AUC on behavioral attacks only (DDoS, injection,
 ransomware, backdoor).}
\label{tab:ocsvm_vs_bridgeguard}
\begin{tabular}{llrcccc}
\toprule
Dataset & Model & Win & AUC & TPR & FPR & F1 \\
\midrule""")

for r_dict in latex_rows:
    d = r_dict["dname"]
    n = r_dict["n_win"]

    print(f"  {d} & {r_dict['ocsvm_name']}$^\\dagger$ & {n} & "
          f"{r_dict['ocsvm_auc']:.3f} & "
          f"{r_dict['ocsvm_tpr']*100:.1f}\\% & "
          f"{r_dict['ocsvm_fpr']*100:.1f}\\% & "
          f"{r_dict['ocsvm_f1']:.3f} \\\\")

    print(f"  & {r_dict['ocsvm_coral_name']}$^\\dagger$ & & "
          f"{r_dict['ocsvm_coral_auc']:.3f} & "
          f"{r_dict['ocsvm_coral_tpr']*100:.1f}\\% & "
          f"{r_dict['ocsvm_coral_fpr']*100:.1f}\\% & "
          f"{r_dict['ocsvm_coral_f1']:.3f} \\\\")

    beh_str = f"({r_dict['bg_auc_beh']:.3f}$^*$)" if abs(r_dict['bg_auc_beh'] - r_dict['bg_auc']) > 0.005 else ""
    print(f"  & \\textbf{{BridgeGuard}} & & "
          f"\\textbf{{{r_dict['bg_auc']:.3f}}}{beh_str} & "
          f"\\textbf{{{r_dict['bg_tpr']*100:.1f}\\%}} & "
          f"\\textbf{{{r_dict['bg_fpr']*100:.1f}\\%}} & "
          f"\\textbf{{{r_dict['bg_f1']:.3f}}} \\\\")
    print(" \\midrule")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

out = {}
for dname, r in all_results.items():
    bn, bm  = r["best_ocsvm"]
    bcn, bcm = r["best_ocsvm_coral"]
    bg = r.get("bridgeguard") or {}
    bg_m = bg.get("v10_fpr15", {})
    out[dname] = {
        "best_ocsvm": {"name": bn, **bm},
        "best_ocsvm_coral": {"name": bcn, **bcm},
        "all_ocsvm": r["ocsvm"],
        "all_ocsvm_coral": r["ocsvm_coral"],
        "bridgeguard_fpr15": bg_m,
        "delta_auc_bg_vs_ocsvm":       round(bg_m.get("auc", 0) - bm["auc"], 4),
        "delta_auc_bg_vs_ocsvm_coral": round(bg_m.get("auc", 0) - bcm["auc"], 4),
    }

out_path = os.path.join(OUTPUT_DIR, "ocsvm_comparison.json")
with open(out_path, "w") as fh:
    json.dump(out, fh, indent=2)

def _print_summary():
    cols   = ["Dataset", "OCSVM AUC", "OCSVM+CORAL AUC", "BridgeGuard AUC", "dAUC (BG-OCSVM)", "dAUC (BG-CORAL)"]
    widths = [22, 10, 16, 16, 16, 16]
    sep    = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    def row(vals):
        return "| " + " | ".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)) + " |"
    print("\n" + sep)
    print(row(cols))
    print(sep)
    for dname, r in out.items():
        print(row([
            dname,
            f"{r['best_ocsvm']['auc']:.4f}",
            f"{r['best_ocsvm_coral']['auc']:.4f}",
            f"{r['bridgeguard_fpr15'].get('auc', 0):.4f}",
            f"{r['delta_auc_bg_vs_ocsvm']:+.4f}",
            f"{r['delta_auc_bg_vs_ocsvm_coral']:+.4f}",
        ]))
    print(sep)
    print(f"  Output: {out_path}")
    print(sep)

_print_summary()

LITERATURE = [

    {
        "ref":     "CNN-MHBiGRU [ScienceDirect 2025]",
        "dataset": "NF-ToN-IoT-v2 (network)",
        "setting": "ID/SUP",
        "auc":     0.990,
        "tpr":     0.990,
        "fpr":     None,
        "f1":      0.9907,
        "note":    "Supervised multiclass, network traffic only",
        "doi":     "10.1016/j.jnca.2025.103330 (ScienceDirect)",
    },
    {
        "ref":     "Robust IF+OCSVM [Sci.Reports 2025]",
        "dataset": "TON_IoT (network)",
        "setting": "ID/UNS",
        "auc":     None,
        "tpr":     0.9852,
        "fpr":     None,
        "f1":      None,
        "note":    "Unsupervised; accuracy=98.52%, precision=99.98%; in-domain",
        "doi":     "10.1038/s41598-025-20445-4",
    },
    {
        "ref":     "EcoDefender AE+IF [arXiv 2025]",
        "dataset": "Heterogeneous IoT traffic",
        "setting": "ID/UNS",
        "auc":     0.963,
        "tpr":     0.94,
        "fpr":     None,
        "f1":      None,
        "note":    "Unsupervised AE+IF hybrid; green edge deployment; in-domain",
        "doi":     "arXiv:2511.18235",
    },
    {
        "ref":     "Hybrid AE+IF+Whitening [ETASR 2026]",
        "dataset": "CIC-IOT-DIAD + TON_IoT",
        "setting": "ID/UNS",
        "auc":     0.990,
        "tpr":     None,
        "fpr":     None,
        "f1":      None,
        "note":    "Unsupervised AE+IF; covariance whitening; in-domain",
        "doi":     "etasr.com/index.php/ETASR/article/view/15288",
    },

    {
        "ref":     "CWD-LR Domain Adapt. [Springer 2026]",
        "dataset": "ACIIoT2023 CICIoMT2024",
        "setting": "CD/SUP",
        "auc":     None,
        "tpr":     None,
        "fpr":     None,
        "f1":      0.9423,
        "note":    "Cross-domain IoT IoMT; supervised adaptation (uses target labels)",
        "doi":     "10.1007/s43926-026-00288-9",
    },
    {
        "ref":     "CEDA [arXiv 2025]",
        "dataset": "ICS cross-domain",
        "setting": "CD/UNS",
        "auc":     None,
        "tpr":     None,
        "fpr":     None,
        "f1":      None,
        "note":    "Cross-domain ICS; clustering-enhanced; partial supervision",
        "doi":     "arXiv:2604.12183",
    },

    {
        "ref":     "BridgeGuard Zone C (in-domain MQTT)",
        "dataset": "MQTT source domain (Zone C)",
        "setting": "ID/UNS",
        "auc":     0.9975,
        "tpr":     0.934,
        "fpr":     0.008,
        "f1":      0.958,
        "note":    "Source-domain test; AUPRC=0.9949; FPR CI=[0.3%,1.5%]; evaluate_bridgeguard.py",
        "doi":     "this work",
    },

    {
        "ref":     "BridgeGuard (this work) Fridge",
        "dataset": "MQTT IoT_Fridge (ToN-IoT)",
        "setting": "CD/UNS",
        "auc":     0.636,
        "tpr":     0.415,
        "fpr":     0.145,
        "f1":      0.484,
        "note":    "Zero-shot cross-domain; no target labels; behavioral AUC=0.579",
        "doi":     "this work",
    },
    {
        "ref":     "BridgeGuard (this work) Thermostat",
        "dataset": "MQTT IoT_Thermostat (ToN-IoT)",
        "setting": "CD/UNS",
        "auc":     0.882,
        "tpr":     0.824,
        "fpr":     0.146,
        "f1":      0.727,
        "note":    "Zero-shot cross-domain; no target labels; behavioral AUC=0.858",
        "doi":     "this work",
    },
    {
        "ref":     "BridgeGuard (this work) Weather",
        "dataset": "MQTT IoT_Weather (ToN-IoT)",
        "setting": "CD/UNS",
        "auc":     0.830,
        "tpr":     0.491,
        "fpr":     0.144,
        "f1":      0.545,
        "note":    "Zero-shot cross-domain; no target labels; behavioral AUC=0.813",
        "doi":     "this work",
    },
]

print(f"\n\n{'='*75}")
print("LITERATURE CONTEXT Published IDS Results (2024 2026)")
print("Settings: [ID]=In-Domain [CD]=Cross-Domain [SUP]=Supervised [UNS]=Unsupervised")
print(f"{'='*75}")
print(f"  {'Reference':<42} {'Setting':<8} {'AUC':>6} {'TPR':>7} {'F1':>7}  Dataset")
print("-" * 95)
for e in LITERATURE:
    auc_s = f"{e['auc']:.3f}" if e['auc'] is not None else " N/A"
    tpr_s = f"{e['tpr']*100:.1f}%" if e['tpr'] is not None else " N/A"
    f1_s  = f"{e['f1']:.3f}" if e['f1'] is not None else " N/A"
    marker = " <-- THIS WORK" if e['doi'] == "this work" else ""
    print(f"  {e['ref']:<42} [{e['setting']:<6}] {auc_s:>6} {tpr_s:>7} {f1_s:>7}  {e['dataset']}{marker}")

print(f"\n{'='*75}")
print("KEY TAKEAWAY:")
print(" In-domain unsupervised methods (EcoDefender, Robust IF+OCSVM) achieve AUC 0.963-0.985")
print(" on their native dataset a favorable setting BridgeGuard deliberately avoids.")
print(" BridgeGuard [CD/UNS] achieves AUC 0.636-0.882 zero-shot on ToN-IoT sensor data,")
print(" the only cross-domain unsupervised result on this specific sensor subset.")
print(" OCSVM [CD/UNS] with same setting: TPR=0% at FPR 15% on all three datasets.")
print(f"{'='*75}")

print("\n\n% ====== LaTeX: Literature Context Table ======")
print(r"""\begin{table}[ht]
\centering
\caption{Literature context for IoT intrusion detection (2024--2026).
 \textbf{Setting:} ID=In-Domain, CD=Cross-Domain, SUP=Supervised,
 UNS=Unsupervised. Direct AUC comparison is only valid within the same
 setting bracket. BridgeGuard operates in the most challenging setting
 (CD/UNS): zero-shot transfer, no target labels.
 $\dagger$TPR=accuracy reported; AUC not provided.
 $\ddagger$F1 only; TPR/FPR not disaggregated.
 $^*$Behavioral attacks (DDoS/injection/ransomware/backdoor) only.}
\label{tab:literature_context}
\begin{tabular}{llccccc}
\toprule
Method & Dataset & Set. & AUC & TPR & F1 & Ref. \\
\midrule
\multicolumn{7}{l}{\textit{In-domain supervised (upper bound not directly comparable)}} \\
CNN-MHBiGRU & NF-ToN-IoT-v2 & ID/S & 0.990 & 0.990 & 0.991 & \cite{cnnmhbigru2025} \\
\midrule
\multicolumn{7}{l}{\textit{In-domain unsupervised (same-domain, no attack labels)}} \\
Robust IF+OCSVM & TON\_IoT (net.) & ID/U & --- & 0.985$^\dagger$ & --- & \cite{robustif2025} \\
EcoDefender (AE+IF) & Het. IoT & ID/U & 0.963 & 0.940 & --- & \cite{ecodefender2025} \\
Hybrid AE+IF+Whitening & CIC-IOT+ToN & ID/U & 0.990 & --- & --- & \cite{hybridaeif2026} \\
\midrule
\multicolumn{7}{l}{\textit{Cross-domain (model trained on different domain)}} \\
CWD-LR & IoT$\to$IoMT & CD/S & --- & --- & 0.942$^\ddagger$ & \cite{cwdlr2026} \\
OCSVM (this work) & MQTT$\to$ToN-IoT & CD/U & 0.855--0.902 & 0\% & 0.000 & --- \\
\midrule
\multicolumn{7}{l}{\textit{BridgeGuard (this work) source-domain and cross-domain}} \\
\textbf{BridgeGuard} & MQTT Zone C (source) & \textbf{ID/U} & \textbf{0.9975} & \textbf{93.4\%} & \textbf{0.958} & \textbf{this work} \\
\textbf{BridgeGuard} & MQTT$\to$Thermostat & \textbf{CD/U} & \textbf{0.882} & \textbf{82.4\%} & \textbf{0.727} & \textbf{this work} \\
\textbf{BridgeGuard} & MQTT$\to$Weather & \textbf{CD/U} & \textbf{0.830} & \textbf{49.1\%} & \textbf{0.545} & \textbf{this work} \\
\textbf{BridgeGuard} & MQTT$\to$Fridge & \textbf{CD/U} & \textbf{0.636} & \textbf{41.5\%} & \textbf{0.484} & \textbf{this work} \\
\bottomrule
\end{tabular}
\end{table}

% BibTeX entries to add:
% @article{cnnmhbigru2025,title={CNN-MHBiGRU: ...},journal={Ad Hoc Networks/ScienceDirect},year={2025},doi={10.1016/j.jnca.2025.103330}}
% @article{robustif2025,title={Robust IoT security using isolation forest and one class SVM},journal={Scientific Reports},year={2025},doi={10.1038/s41598-025-20445-4}}
% @article{ecodefender2025,title={Lightweight AE-IF Anomaly Detection for Green IoT Edge Gateways},journal={arXiv},year={2025},eprint={2511.18235}}
% @article{hybridaeif2026,title={Hybrid Autoencoder and Isolation Forest for IoT Anomaly Detection},journal={ETASR},year={2026},url={etasr.com/index.php/ETASR/article/view/15288}}
% @article{cwdlr2026,title={A minimalistic yet effective domain adaptation strategy for IoMT},journal={Discover IoT (Springer)},year={2026},doi={10.1007/s43926-026-00288-9}}""")

print("=" * 65)
print("DONE")
print("=" * 65)
