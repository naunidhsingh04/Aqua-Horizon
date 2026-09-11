# 🌊 AQUA HORIZON
### AI-Driven 7-Day Inundation Forecasting & Disaster Intelligence Platform for India

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/ML-ADASYN%20%2B%20XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Leaflet](https://img.shields.io/badge/GIS-Leaflet.js-green.svg)](https://leafletjs.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## 📌 Overview

**AQUA HORIZON** is an end-to-end AI disaster management platform designed for district-level flood prediction, risk assessment, and early public warning across India. Grounded in the **57-year national IFI-Impacts database (1967–2023)** and connected directly to **live GFS satellite and meteorological radar feeds**, AQUA HORIZON predicts inundation probabilities 7 days in advance across **726 Indian districts**.

---

## 🚀 Key Features

* **57-Year Historical Grounding:** Trained on the comprehensive *India Flood Inventory–Impacts (IFI-Impacts 1967–2023)* dataset from Zenodo, covering 41,325 district-year observations.
* **Strict Zero Target Leakage:** Strictly excludes post-disaster consequence metrics (fatalities, livestock loss, crop damage) from the occurrence classifier. Rolling historical features are calculated using only prior years ($t-1$).
* **Chronological 80:20 Ratio Split:** Model trained strictly on **1967–2012 (80.7%)** and tested on untouched **2013–2023 (19.3%)** data to reflect genuine real-world disaster forecasting.
* **ADASYN Class Imbalance Handling:** Solves severe class imbalance (~1:2.5 disaster frequency) by synthesizing minority flood samples strictly within the training set, boosting test recall from **45.3% to 66.9%**.
* **Live GFS Satellite Feeds:** Integrates high-resolution **Open-Meteo GFS weather telemetry**, providing 7-day daily precipitation (mm) and hydrological soil wetness indices across India.
* **Interactive Fullscreen Geospatial Command Center:**
  * Clean, dark-mode Leaflet choropleth tracking **726 districts** with zero clutter.
  * Instant cursor hover tooltips displaying district name, state, risk %, and rain mm.
  * 7-day horizontal outlook dock (`Today`, `Tomorrow`, `Day 3` ... `Day 7`).
  * Free zero-API-key Esri Satellite and Dark Canvas toggles.
* **Instant Citizen Advisory Dispatch (WhatsApp & SMS Ready):**
  * Auto-drafts concise, non-technical public warning messages tailored for NDRF, District Magistrates, and citizen communication.
  * One-click **Copy Message** and direct **Open in WhatsApp** broadcast integration.
  * Highlights 24x7 District Emergency Helpline: `1077`.

---

## 🔬 Scientific ML Benchmarks (Chronological Test Partition: 2013–2023)

| Model Configuration | Recall | Precision | F1-Score | PR-AUC | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Control - Raw Data)** | 45.3% | **64.4%** | 0.532 | 0.657 | 0.789 | Under-predicts rare floods |
| **SMOTE Balanced** | 57.5% | 61.3% | 0.593 | 0.665 | 0.793 | +12.2% Recall improvement |
| **ADASYN Balanced (Champion)** | **66.9%** | 58.2% | **0.615** | **0.667** | **0.796** | **Highest Safety & Detection (+21.6% Recall)** |

> In disaster prediction, **Recall** is paramount: missing a flood (false negative) causes human casualties, whereas early warnings enable life-saving evacuations.

---

## 📂 Project Structure

```text
├── data/
│   ├── raw/                             # Raw Zenodo IFI-Impacts CSV files
│   ├── processed/                       # 41,325-row District-Year panel dataset
│   └── india_districts.geojson          # Clean 726-district boundary polygons
├── ml/
│   ├── 01_preprocess.py                 # Multi-district explosion & zero-leakage feature engineering
│   ├── 02_train_models.py               # 80:20 chronological training, SMOTE, ADASYN & XGBoost
│   ├── 03_export_web_data.py            # Live Open-Meteo GFS telemetry ingestion & web export
│   └── models/                          # Trained models (.joblib) & benchmark metrics
├── web/
│   ├── index.html                       # Modern command center interface
│   ├── css/style.css                    # Dark glassmorphic styling & responsive dock
│   ├── js/app.js                        # Leaflet map logic, hover tooltips & advisory engine
│   └── data/                            # Enriched GeoJSON & 7-day live telemetry payloads
├── predict.py                           # Standalone CLI flood risk predictor
├── run_app.py                           # Threaded local web server
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites & Dependencies
```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn joblib requests
```

### 2. Launch the Web Application
```bash
python run_app.py
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

### 3. CLI Prediction Tool
Predict live or hypothetical scenario flood risk for any district:
```bash
# Predict live risk for Patna
python predict.py --district Patna

# Predict with custom rainfall scenario (+35mm rain)
python predict.py --district Wayanad --rain 35.0
```

---

## 👥 Authors & Acknowledgments

* **IFI-Impacts Dataset (1967–2023):** Hosted on [Zenodo (Record 11275211)](https://zenodo.org/records/11275211).
* **Live Meteorology:** Open-Meteo High-Resolution GFS Weather Feed.
* **Map Tiles:** Esri World Imagery & Dark Gray Canvas.