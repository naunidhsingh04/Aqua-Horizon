# 🌊 AQUA HORIZON
### AI-Driven 7-Day Inundation Forecasting & Disaster Intelligence Platform for India

[![Live Demo](https://img.shields.io/badge/Live_Demo-aqua--horizon.vercel.app-00dfa2?style=for-the-badge&logo=vercel&logoColor=white)](https://aqua-horizon.vercel.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/ML-ADASYN%20%2B%20XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Leaflet](https://img.shields.io/badge/GIS-Leaflet.js-green.svg)](https://leafletjs.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> 🚀 **Live Production Deployment:** [**https://aqua-horizon.vercel.app/**](https://aqua-horizon.vercel.app/)

---

## 📌 Overview

**AQUA HORIZON** is an end-to-end AI disaster management platform designed for district-level flood prediction, flood severity mapping, and early public warning across India. Grounded in the **57-year national IFI-Impacts database (1967–2023)** and connected directly to **live GFS satellite and meteorological radar feeds**, AQUA HORIZON predicts inundation probabilities 7 days in advance across **726 Indian districts**.

In compliance with the official **Disaster Management AI Challenge**, AQUA HORIZON implements and benchmarks **5 Hybrid Deep Learning Techniques** alongside statistical cross-validation ensembles:
1. **U-Net + ConvLSTM** (Spatial-Temporal Inundation Propagation)
2. **CNN + LSTM** (1D Temporal Feature Extractor + Recurrent Memory)
3. **CNN + Transformer** (Multi-Head Self-Attention Temporal Network)
4. **ResNet + BiLSTM** (Deep Residual Skip Connections + Bidirectional Sequence Modeling)
5. **Attention U-Net + LSTM** (Additive Attention-Gated Basin Hydrology + Recurrent Cell)

---

## 🧠 5 Hybrid Deep Learning Architectures (Hackathon Mandate)

All 5 hybrid architectures are implemented in modular PyTorch files in `ml/` and can be trained individually or via the master benchmark orchestrator:

| Model Architecture | Parameters | Test Recall | Test Precision | F1-Score | ROC-AUC | PR-AUC | Dedicated Training Script | Weights Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Attention U-Net + LSTM** | 26,196 | **82.08%** | 52.56% | 0.6408 | 0.8017 | 0.6713 | `ml/train_05_attention_unet_lstm.py` | `ml/models/attention_unet_lstm.pt` |
| **U-Net + ConvLSTM** | 85,729 | 81.86% | 52.28% | 0.6381 | 0.7996 | 0.6717 | `ml/train_01_unet_convlstm.py` | `ml/models/unet_convlstm.pt` |
| **CNN + Transformer** | 70,849 | 81.15% | 52.39% | 0.6367 | 0.7946 | 0.6537 | `ml/train_03_cnn_transformer.py` | `ml/models/cnn_transformer.pt` |
| **CNN + LSTM** | 67,889 | 80.63% | 53.20% | **0.6410** | **0.8058** | **0.6766** | `ml/train_02_cnn_lstm.py` | `ml/models/cnn_lstm.pt` |
| **ResNet + BiLSTM** | 81,121 | 75.33% | **54.84%** | 0.6347 | 0.7985 | 0.6638 | `ml/train_04_resnet_bilstm.py` | `ml/models/resnet_bilstm.pt` |

### Architecture Details:
* **`UNetConvLSTM`**: Integrates a 2D spatial U-Net encoder-decoder with skip connections and a `ConvLSTMCell` at the bottleneck. Converts $15$ hydrological features into a $4\times 4$ spatial latent grid over $T=5$ temporal sequence steps to model inundation propagation.
* **`CNNLSTM`**: Employs 1D temporal convolutions for feature projection and local pattern detection, fed into a 2-layer LSTM sequential memory network.
* **`CNNTransformer`**: Projects multi-year sequence features and feeds them into a 4-head Transformer Encoder with GELU activations and positional encodings to capture long-range historical climate cycles.
* **`ResNetBiLSTM`**: Features 1D residual blocks with identity skip connections to prevent vanishing gradients, coupled with a bidirectional LSTM processing forward hydrological trends and backward baseline context.
* **`AttentionUNetLSTM`**: Incorporates additive attention gates ($\psi, 	heta_x, \phi_g$) to selectively focus on low-lying drainage basins and flash flood zones before LSTM sequence prediction.

---

## 🔬 Statistical Validation: 5-Fold Stratified Cross-Validation & Ensemble

To ensure scientific rigor and prevent single-split variance, AQUA HORIZON was also evaluated using **5-Fold Stratified Cross-Validation** across all 41,325 district-year observations:

| Model Configuration | Recall | Precision | F1-Score | PR-AUC | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Control - Raw Data)** | 46.4% | **65.2%** | 0.542 | 0.666 | 0.796 | Under-predicts rare floods |
| **SMOTE Balanced** | 61.3% | 60.6% | 0.609 | **0.669** | 0.796 | +14.9% Recall improvement |
| **ADASYN Balanced (Holdout)** | 60.1% | 60.5% | 0.603 | 0.667 | 0.796 | +13.7% Recall improvement |
| **5-Fold Stratified CV (ADASYN + XGBoost)** | **66.7% (±0.8%)** | 54.9% (±0.4%) | **0.602** | 0.633 | **0.808** | **Statistically Verified Champion (OOF Ensemble)** |

> In flood disaster management, **Recall is life-critical**: false negatives mean unprepared communities, whereas early alerts enable pre-emptive disaster mitigation and evacuation.

---

## 🚀 Key Features of the Live Platform

* **Real-Time Architecture Switching:** Live UI allows users/judges to toggle between **all 5 Hybrid Deep Learning models** and the **Ensemble** via the top navigation bar.
* **57-Year Historical Grounding:** Trained on the national *India Flood Inventory–Impacts (IFI-Impacts 1967–2023)* dataset from Zenodo (41,325 district-year records).
* **Strict Zero Target Leakage:** Strictly excludes post-disaster consequence metrics (fatalities, damage) from prediction features.
* **Live GFS Satellite Feeds:** Ingests live high-resolution **Open-Meteo GFS weather telemetry** with 7-day precipitation forecasts.
* **Interactive Fullscreen Geospatial Command Center:**
  * Clean, dark-mode Leaflet choropleth tracking **726 districts** with zero visual clutter.
  * Instant cursor hover tooltips displaying district name, state, risk %, and rain mm.
  * 7-day horizontal outlook dock (`Today`, `Tomorrow`, `Day 3` ... `Day 7`).
  * Free zero-API-key Esri Satellite and Dark Canvas toggles.
* **Instant Citizen Advisory Dispatch (WhatsApp & SMS Ready):**
  * Auto-drafts concise, non-technical public warning messages tailored for NDRF, District Magistrates, and citizen communication.
  * One-click **Copy Message** and direct **Open in WhatsApp** broadcast integration.
  * Highlights 24x7 District Emergency Helpline: `1077`.

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
│   ├── models_hybrid.py                 # 5 PyTorch Hybrid Deep Learning Architectures
│   ├── data_loader.py                   # Multi-year sequence builder (T=5)
│   ├── train_01_unet_convlstm.py        # Model 1: U-Net + ConvLSTM
│   ├── train_02_cnn_lstm.py             # Model 2: CNN + LSTM
│   ├── train_03_cnn_transformer.py      # Model 3: CNN + Transformer
│   ├── train_04_resnet_bilstm.py        # Model 4: ResNet + BiLSTM
│   ├── train_05_attention_unet_lstm.py  # Model 5: Attention U-Net + LSTM
│   ├── train_all_hybrid_models.py       # Master Training & Benchmark Orchestrator
│   ├── ensemble.py                      # K-Fold Soft-Voting Ensemble Classifier
│   └── models/                          # Saved PyTorch weights (.pt) & JSON benchmarks
├── web/
│   ├── index.html                       # Modern command center interface
│   ├── css/style.css                    # Dark glassmorphic styling & responsive dock
│   ├── js/app.js                        # Leaflet map logic, hover tooltips & hybrid switcher
│   └── data/                            # GeoJSON, intelligence payloads & hybrid benchmarks
├── predict.py                           # Standalone CLI flood risk predictor
├── run_app.py                           # Threaded local web server
├── vercel.json                          # Production deployment config
└── README.md
```

---

## ⚡ Quick Start

### 1. Prerequisites & Dependencies
```bash
pip install torch numpy pandas scikit-learn xgboost imbalanced-learn joblib requests
```

### 2. Run Master Hybrid Model Suite
To train and benchmark all 5 hybrid deep learning architectures:
```bash
python ml/train_all_hybrid_models.py
```
Or train individual architectures:
```bash
python ml/train_01_unet_convlstm.py
python ml/train_02_cnn_lstm.py
python ml/train_03_cnn_transformer.py
python ml/train_04_resnet_bilstm.py
python ml/train_05_attention_unet_lstm.py
```

### 3. Access the Web Application

* **Live Production Deployment (Instant Access):**  
  👉 [**https://aqua-horizon.vercel.app/**](https://aqua-horizon.vercel.app/)

* **Or Run Locally:**  
  ```bash
  python run_app.py
  ```
  Open your browser at: `http://localhost:8000`

### 4. CLI Prediction Tool
Predict live or hypothetical scenario flood risk for any district:
```bash
# Predict live risk for Patna
python predict.py --district Patna

# Predict with custom rainfall scenario (+35mm rain)
python predict.py --district Dibrugarh --rain 35.0
```

---

## 👥 Authors & License

Developed for the **AI Hackathon 2026** — Problem Statement 2: *AI-Based Flood Prediction and Flood Severity Mapping*.  
Licensed under the [MIT License](LICENSE).
