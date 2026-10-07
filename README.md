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

| Model Architecture | Parameters | Test Accuracy | Test Recall | Test Precision | F1-Score | ROC-AUC | PR-AUC | Dedicated Training Script | Weights Path |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CNN + Transformer** | 70,849 | 69.88% | 78.91% | 53.51% | 0.6377 | 0.7963 | 0.6404 | `ml/train_03_cnn_transformer.py` | `ml/models/cnn_transformer.pt` |
| **U-Net + ConvLSTM** | 85,729 | 69.33% | 79.99% | 52.87% | 0.6367 | 0.7946 | 0.6490 | `ml/train_01_unet_convlstm.py` | `ml/models/unet_convlstm.pt` |
| **ResNet + BiLSTM** | 81,121 | 69.25% | 79.10% | 52.83% | 0.6335 | 0.7979 | 0.6689 | `ml/train_04_resnet_bilstm.py` | `ml/models/resnet_bilstm.pt` |
| **Attention U-Net + LSTM** | 26,196 | 68.88% | 83.17% | 52.31% | 0.6423 | 0.7995 | 0.6475 | `ml/train_05_attention_unet_lstm.py` | `ml/models/attention_unet_lstm.pt` |
| **CNN + LSTM** | 67,889 | 67.15% | 84.14% | 50.66% | 0.6324 | 0.7956 | 0.6625 | `ml/train_02_cnn_lstm.py` | `ml/models/cnn_lstm.pt` |

### Architecture Details:
* **`UNetConvLSTM`**: Integrates a 2D spatial U-Net encoder-decoder with skip connections and a `ConvLSTMCell` at the bottleneck. Converts $15$ hydrological features into a $4\times 4$ spatial latent grid over $T=5$ temporal sequence steps to model inundation propagation.
* **`CNNLSTM`**: Employs 1D temporal convolutions for feature projection and local pattern detection, fed into a 2-layer LSTM sequential memory network.
* **`CNNTransformer`**: Projects multi-year sequence features and feeds them into a 4-head Transformer Encoder with GELU activations and positional encodings to capture long-range historical climate cycles.
* **`ResNetBiLSTM`**: Features 1D residual blocks with identity skip connections to prevent vanishing gradients, coupled with a bidirectional LSTM processing forward hydrological trends and backward baseline context.
* **`AttentionUNetLSTM`**: Incorporates additive attention gates ($\psi, \theta_x, \phi_g$) to selectively focus on low-lying drainage basins and flash flood zones before LSTM sequence prediction.

---

## 🔬 Statistical Validation: 10-Fold Stratified Cross-Validation & Ensemble

### A. 10-Fold Stratified Cross-Validation for 5 Hybrid Deep Learning Models

To verify cross-geographical stability and eliminate fold-specific bias, all 5 hybrid deep learning architectures were trained and validated across **10 Stratified Folds (38,425 multi-year sequences)** using training-fold-only standardizers:

| Hybrid Model Architecture | 10-Fold Accuracy (±Std) | 10-Fold Mean Recall (±Std) | 10-Fold Mean Precision (±Std) | Mean F1-Score | Mean ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN + LSTM** | 70.59% (±1.56%) | 75.16% (±2.63%) | 49.61% (±1.83%) | 0.5971 | 0.7980 |
| **ResNet + BiLSTM** | 70.03% (±1.95%) | 78.01% (±3.88%) | 49.08% (±2.09%) | 0.6016 | 0.8024 |
| **U-Net + ConvLSTM** | 69.39% (±1.59%) | 77.00% (±4.17%) | 48.33% (±1.57%) | 0.5931 | 0.7967 |
| **CNN + Transformer** | 69.00% (±1.04%) | 77.84% (±2.68%) | 47.89% (±1.03%) | 0.5927 | 0.7936 |
| **Attention U-Net + LSTM** | 68.99% (±2.76%) | 75.66% (±5.47%) | 48.09% (±2.66%) | 0.5860 | 0.7878 |

*Run 10-Fold Hybrid Suite:*
```bash
python ml/train_kfold_hybrid_models.py
```

> The 10-fold figures above use random folds over overlapping 5-year windows of the same districts, so they can look somewhat better than a forecast of future years would. The 80:20 table (split by year) is the stricter test.

#### Top-3 models and their combinations

The three best single models (ResNet + BiLSTM, CNN + LSTM, CNN + Transformer) are combined by averaging their flood probabilities: first two, last two, first + last, and all three. All seven rows are scored the same way: SMOTE on the training data only, the decision threshold chosen on a validation slice, and the 80:20 split by year and the 10-fold cross-validation both measured on real (unbalanced) test data.

| Model / Combination | 80:20 Accuracy | 10-Fold Accuracy (±Std) | Recall | Precision | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet + BiLSTM** | 71.6% | 71.5% (±1.8%) | 72.2% | 50.8% | 0.595 | 0.800 |
| **CNN + LSTM** | 71.7% | 69.8% (±1.9%) | 75.4% | 48.8% | 0.591 | 0.798 |
| **CNN + Transformer** | 72.4% | 69.9% (±1.6%) | 74.5% | 48.9% | 0.589 | 0.794 |
| **ResNet+BiLSTM + CNN+LSTM** | 70.2% | 71.2% (±1.5%) | 73.7% | 50.3% | 0.597 | 0.803 |
| **CNN+LSTM + CNN+Transformer** | 72.2% | 70.0% (±1.4%) | 76.0% | 48.9% | 0.595 | 0.799 |
| **ResNet+BiLSTM + CNN+Transformer** | 72.4% | 71.0% (±1.7%) | 74.4% | 50.1% | 0.598 | 0.803 |
| **ResNet+BiLSTM + CNN+LSTM + CNN+Transformer** | 72.6% | 70.8% (±2.1%) | 75.0% | 49.9% | 0.598 | 0.803 |

The combinations rank flood risk slightly better (ROC-AUC) than any single model, but the accuracy differences are 1 to 2 points, which is within the fold-to-fold spread (the ± value). *Run:*
```bash
python ml/train_combo_hybrid_models.py
```

### B. Tabular Baselines & Resampling Strategies (Control vs SMOTE vs ADASYN vs Ensemble)

To ensure scientific rigor and prevent single-split variance, AQUA HORIZON was also evaluated using **10-Fold Stratified Cross-Validation** across all 41,325 district-year observations:

| Model Configuration | 80:20 Accuracy | 10-Fold Accuracy (±Std) | Recall | Precision | F1-Score | PR-AUC | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Control - Raw)** | 73.7% | 76.4% (±0.6%) | 57.8% | 59.4% | 0.586 | 0.638 | 0.810 |
| **SMOTE Balanced** | 73.6% | 75.0% (±0.7%) | 66.1% | 55.5% | 0.603 | 0.634 | 0.807 |
| **ADASYN Balanced** | 73.4% | 74.9% (±0.6%) | 66.9% | 55.3% | 0.605 | 0.635 | 0.809 |

> In flood disaster management, **Recall is life-critical**: false negatives mean unprepared communities, whereas early alerts enable pre-emptive disaster mitigation and evacuation.

### C. 10-Fold Inundation Risk & Satellite Severity Benchmarks

> **Note:** this benchmark uses the already-balanced SMOTE dataset, so synthetic rows can appear in both the training and test folds. Treat these figures as an upper bound; the stricter results (SMOTE on training data only) are in `ml/train_combo_hybrid_models.py`.

To maximize output fidelity and eliminate single-split bias, our 58,850-row balanced dataset was benchmarked using a rigorous **10-Fold Stratified Cross-Validation** process testing both flood classification and continuous satellite inundation extent (MODIS Global Flood Database):

| Metric | 10-Fold Mean | Standard Deviation (±Std) | Note |
| :--- | :---: | :---: | :--- |
| **Classification Accuracy** | 83.10% | ±0.46% | Optimistic: SMOTE rows can sit in both train and test folds |
| **Recall** | 77.88% | ±0.70% | Share of real floods detected |
| **Precision** | 86.96% | ±0.72% | Share of flood alerts that were correct |
| **F1-Score** | 0.8217 | ±0.0049 | Balance of recall and precision |
| **ROC-AUC** | 0.9146 | ±0.0020 | Separates flood from non-flood |
| **Satellite Extent R²** | 99.19% | ±0.06% | Inflated: dfsi_score and drainage_stress_ratio are computed from the flooded-area value itself |
| **Satellite Extent MAE** | 0.2180% | | Inflated for the same reason |
| **IRS Rank Correlation (ρ)** | 0.7717 | ±0.0063 | Inundation Risk Score (P × severity) vs the real flooded area |

*Run 10-Fold IRS Suite:*
```bash
python ml/test_10fold_irs.py
```

### D. Satellite model tested in other countries (trained on India only)

The Sentinel-1 flood classifier is trained on India chips only, then run unchanged on Sen1Floods11 chips from the USA, Pakistan and Sri Lanka. Nothing from those countries is used for training or for choosing the best epoch. Only these four events are tested; Nepal, Bhutan and Myanmar are not in Sen1Floods11.

| Test set | Chips | Flood chips | Accuracy | Recall | Precision | F1 | ROC-AUC | Accuracy of always saying "no flood" |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **India (held-out 20%, same country)** | 240 | 31.7% | 82.9% | 69.7% | 74.7% | 0.721 | 0.863 | 68.3% |
| **USA** | 4289 | 10.4% | 42.9% | 91.3% | 14.5% | 0.250 | 0.803 | 89.5% |
| **Pakistan** | 1130 | 14.6% | 53.3% | 62.4% | 18.1% | 0.281 | 0.632 | 85.4% |
| **Sri-Lanka** | 2237 | 20.4% | 84.9% | 37.1% | 77.2% | 0.501 | 0.865 | 79.6% |

How to read this: the model works in Sri Lanka (accuracy 84.9%, ROC-AUC 0.865) but misses many floods there (recall 37%). In the USA and Pakistan it raises many false alarms, so its accuracy is below simply always answering "no flood". The USA still ranks flood chips well (AUC 0.803), so a different decision threshold would help there; Pakistan ranks poorly (AUC 0.632). One India-only model does not carry over to every country. *Run:*
```bash
python ml/download_sen1floods11_india.py India Pakistan Sri-Lanka USA
python ml/test_other_countries.py
```
Outputs: `ml/models/other_countries_metrics.json`, `ml/models/other_countries_predictions.csv` (one row per chip) and sample pictures in `ml/models/other_countries_samples/`. The website is not changed by this test.

---

## 🚀 Key Features of the Live Platform

* **Real-Time Architecture Switching:** Live UI allows users/judges to toggle between **all 5 Hybrid Deep Learning models** and the **Ensemble** via the top navigation bar.
* **57-Year Historical Grounding:** Trained on the national *India Flood Inventory–Impacts (IFI-Impacts 1967–2023)* dataset from Zenodo (41,325 district-year records).
* **History-only flood features:** the flood-history inputs (previous 3, 5 and 10 years, acceleration) use only years before the one being predicted. The static district severity score (DFSI) is built from the whole 1967-2023 impact record (including fatalities and injuries), so it carries some information about outcomes; treat the scores accordingly.
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
│   ├── raw/                             # Raw Zenodo IFI-Impacts CSV files (+ Sen1Floods11 images, downloaded by script)
│   ├── processed/                       # 41,325-row District-Year panel dataset (created by 01_preprocess.py)
│   └── india_districts.geojson          # Clean 726-district boundary polygons
├── ml/
│   ├── 01_preprocess.py                 # Multi-district explosion & feature engineering (DFSI, flood history)
│   ├── 02_train_models.py               # XGBoost district model: SMOTE / ADASYN, 10-fold + split by year
│   ├── 03_export_web_data.py            # Live Open-Meteo rain forecast + model outputs -> web/data/*.json
│   ├── 04_final_report.py               # One consolidated report of every file's output
│   ├── balance_dataset_smote.py         # 50:50 SMOTE dataset balancing pipeline
│   ├── test_satellite_learning.py       # District features vs flooded-area % (10-fold)
│   ├── test_10fold_irs.py               # 10-Fold Inundation Risk Score (IRS) cross-validation
│   ├── models_hybrid.py                 # The 5 PyTorch hybrid deep-learning architectures (self-test inside)
│   ├── data_loader.py                   # Multi-year sequence builder (T=5) (self-test inside)
│   ├── ensemble.py                      # K-Fold soft-voting ensemble classifier (self-test inside)
│   ├── train_01_unet_convlstm.py        # Model 1: U-Net + ConvLSTM
│   ├── train_02_cnn_lstm.py             # Model 2: CNN + LSTM
│   ├── train_03_cnn_transformer.py      # Model 3: CNN + Transformer
│   ├── train_04_resnet_bilstm.py        # Model 4: ResNet + BiLSTM
│   ├── train_05_attention_unet_lstm.py  # Model 5: Attention U-Net + LSTM
│   ├── train_all_hybrid_models.py       # Runs train_01 to train_05 and writes the comparison table
│   ├── train_kfold_hybrid_models.py     # 10-fold validation of the 5 models + per-district probabilities
│   ├── train_combo_hybrid_models.py     # Top-3 models and their 4 combinations (averaged probabilities)
│   ├── download_sen1floods11_india.py   # Downloads the real Sentinel-1 flood scenes
│   ├── image_pipeline_india.py          # Satellite images: checks, CLAHE, GAN, 10-fold CV
│   ├── test_other_countries.py          # India-trained satellite model tested on USA, Pakistan, Sri Lanka
│   └── models/                          # Saved weights (.pt), model files and JSON / CSV results
├── web/
│   ├── index.html                       # Command center interface
│   ├── css/style.css                    # Dark glass styling, themed scrollbars
│   ├── js/app.js                        # Map logic, model switcher, model-data tables
│   └── data/                            # GeoJSON, intelligence payloads & benchmark JSON (created by the scripts)
├── AquaHorizon_Colab.ipynb              # Runs every file in order on Google Colab
├── requirements.txt                     # Python packages
├── predict.py                           # CLI flood-risk predictor for one district
├── run_app.py                           # Web server (local, or inside a Colab notebook)
├── vercel.json                          # Production deployment config
└── README.md
```

---

## ⚡ Quick Start

### 0. Run everything on Google Colab (recommended for verification)
1. Open [Google Colab](https://colab.research.google.com), choose **File > Upload notebook**, and upload `AquaHorizon_Colab.ipynb`.
2. Run the first code cell and select `AquaHorizon.zip` (the project folder, zipped) when asked.
3. Run the remaining cells in order. Every Python file runs and prints its own output; step 12 prints the final report and step 13 shows the website.

### 1. Prerequisites & Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the whole pipeline in order
```bash
python ml/01_preprocess.py                 # dataset
python ml/balance_dataset_smote.py         # 50:50 balanced copy
python ml/02_train_models.py               # XGBoost district model
python ml/train_all_hybrid_models.py       # the 5 deep-learning models (runs train_01 to train_05)
python ml/train_kfold_hybrid_models.py     # 10-fold validation of the 5 models
python ml/train_combo_hybrid_models.py     # top-3 models and their combinations
python ml/test_10fold_irs.py               # Inundation Risk Score
python ml/test_satellite_learning.py
python ml/download_sen1floods11_india.py   # satellite images (needs internet)
python ml/image_pipeline_india.py
python ml/test_other_countries.py       # India-trained satellite model on USA, Pakistan, Sri Lanka
python ml/03_export_web_data.py            # website data (needs internet for the rain forecast)
python ml/04_final_report.py               # consolidated report
```
Or train one architecture on its own, for example `python ml/train_02_cnn_lstm.py`.

### 3. Access the Web Application

* **Live Production Deployment (Instant Access):**  
  👉 [**https://aqua-horizon.vercel.app/**](https://aqua-horizon.vercel.app/)

* **Or Run Locally:**  
  ```bash
  python run_app.py
  ```
  Open your browser at: `http://localhost:8000`

### 4. CLI Prediction Tool
Flood probability for one district under four what-if rainfall scenarios:
```bash
python predict.py Patna
python predict.py Dibrugarh
```

---

## 👥 Authors & License

Developed for the **AI Hackathon 2026** — Problem Statement 2: *AI-Based Flood Prediction and Flood Severity Mapping*.  
Licensed under the [MIT License](LICENSE).
