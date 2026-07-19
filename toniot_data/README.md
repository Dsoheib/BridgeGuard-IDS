# ToN-IoT Cross-Domain Datasets — Provenance and Integrity

The cross-domain evaluation uses **three unmodified files** from the official
**ToN-IoT** dataset (Alsaedi et al., *IEEE Access*, 2020,
doi:10.1109/ACCESS.2020.3022062), published by UNSW Canberra.

## Official download (direct link)

- Official project page: <https://research.unsw.edu.au/projects/toniot-datasets>
- Official download folder (UNSW SharePoint):
  <https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EvBTaetotpdGnW7rJQ8fCvYBh8063CNeY9W33MpRsarJaQ?e=yZlnxW>
- Inside the folder, navigate to: `Processed_datasets/Processed_IoT_dataset/`

Download the three files below and place them in this directory
(`toniot_data/`), **without any modification**.

## Files used and SHA-256 checksums

| File | SHA-256 | Lines (incl. header) |
|---|---|---|
| `IoT_Fridge.csv` | `e5c7fd42c1d44898eb8afe6800d7a335a38a6a8394c6af083407b9f581736272` | 587,077 |
| `IoT_Thermostat.csv` | `abb919eb3816d30036ad54bfd65bbbdf1e7b527bdb2993c22d9a36a21aa3a5c2` | 442,229 |
| `IoT_Weather.csv` | `1b5e379011d37a1b4bf3598f3c06eb554b207a30787ddcb6a30833092c6ca9a1` | 650,243 |

Verify after download:

```bash
shasum -a 256 -c <<'EOF'
e5c7fd42c1d44898eb8afe6800d7a335a38a6a8394c6af083407b9f581736272  IoT_Fridge.csv
abb919eb3816d30036ad54bfd65bbbdf1e7b527bdb2993c22d9a36a21aa3a5c2  IoT_Thermostat.csv
1b5e379011d37a1b4bf3598f3c06eb554b207a30787ddcb6a30833092c6ca9a1  IoT_Weather.csv
EOF
```

## Integrity statement

The raw CSV files are consumed **as distributed** by UNSW: no rows are
added, removed, relabeled, or augmented. All preprocessing (behavioral
windowing, feature extraction, CORAL alignment, normalization) is performed
**at runtime** by the released scripts
(`cross_domain/cross_domain_evaluation.py`,
`cross_domain/STEP_TONIOT_ADAPTATION.py`) and is therefore fully auditable
and reproducible from the raw files referenced above. The evaluation is
zero-shot: no ToN-IoT window, labeled or unlabeled, is used for model
fitting.
