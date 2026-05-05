# MPS Anomaly Detection — Indus-2 Synchrotron

> **Deep Autoencoder-Based Anomaly Detection in Magnet Power Supply Systems**  
> BTech Final Year Project · RRCAT (Raja Ramanna Centre for Advanced Technology)

An interactive Streamlit dashboard that reproduces the full anomaly detection pipeline from the project report — from raw MPS signal ingestion to fault classification and results export.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [App Pages](#app-pages)
- [Training on Google Colab](#training-on-google-colab)
- [Model Export & Import](#model-export--import)
- [Deployment (Streamlit Community Cloud)](#deployment-streamlit-community-cloud)
- [Technical Reference](#technical-reference)

---

## Overview

The Indus-2 synchrotron at RRCAT operates 117 Magnet Power Supply (MPS) units. Each unit continuously reports two signals:

| Signal | Description |
|---|---|
| `vmeset` | Voltage/current setpoint commanded by the control system |
| `readback` | Actual measured output of the supply |

A healthy supply tracks its setpoint closely. The **deviation** (`readback − vmeset`) should remain near zero. This app trains a deep autoencoder on normal operational data and flags timestamps where reconstruction error exceeds a learned threshold — indicating anomalous MPS behaviour.

**Reported model performance:**

| Metric | Value |
|---|---|
| Accuracy | 95.2% |
| Precision | 93.8% |
| Recall | 94.6% |
| F1-Score | 94.2% |
| Anomaly Threshold (t_mps) | ~0.04 |

---

## Architecture

### Autoencoder

The model follows a symmetric encoder-decoder structure with a 16-dimensional bottleneck:

```
Input (118) → Encoder 64 → Encoder 32 → Latent 16 → Decoder 32 → Decoder 64 → Output (117)
```

| Parameter | Value |
|---|---|
| Loss function | MAE (Mean Absolute Error) |
| Optimiser | Adam |
| Epochs | 25 (local) / 50 with EarlyStopping (Colab) |
| Batch size | 256 (local) / 512 (Colab GPU) |
| Train/Val split | 80 / 20 |

### Anomaly Threshold

```
t_mps = μ + 2σ
```

Where μ and σ are the mean and standard deviation of per-timestamp reconstruction errors on the training set. Timestamps with error > t_mps are flagged as anomalous.

### Beam Cycle Extraction

A state machine scans the event log for valid operational windows:

```
Ramp End ("Ramp done") → [beam on] → Kill Signal ("Kill") → [cycle complete]
```

Cycles that do not form a matched Ramp End → Kill pair are discarded.

### Beam Fault Classification

| Class | Condition |
|---|---|
| No Loss (NL) | Beam current ≥ 1 mA, no sudden drop |
| Partial Loss (PL) | Sudden drop > 2 mA but current > 0 |
| Complete Loss (CL) | Beam current < 1 mA |

### MPS Fault Categories

| Category | Description |
|---|---|
| Single-Supply Single Occurrence | One supply, one anomalous event |
| Single-Supply Repeated | Same supply flags across multiple cycles |
| Multi-Supply Single Occurrence | Several supplies anomalous simultaneously |
| Multi-Supply Repeated | Systemic fault pattern across multiple cycles |

---

## Project Structure

```
mps-anomaly-app/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── MPS_Autoencoder_Colab.ipynb   # Colab notebook for GPU training
├── .streamlit/
│   └── config.toml               # Theme and server configuration
└── README.md
```

---

## Setup & Installation

### Requirements

- Python 3.10 (required — TensorFlow does not yet support 3.12+)
- pip

### Steps

```bash
# 1. Clone or copy the project folder
cd mps-anomaly-app

# 2. Create a virtual environment
python3.10 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**

```
streamlit>=1.28.0
tensorflow>=2.13.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
pyarrow>=12.0.0
```

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default.

> **Upload size limit:** configured to 500 MB in `.streamlit/config.toml`.  
> MPS signal files exported as Parquet are ~8 MB; as CSV they can reach ~220 MB — prefer Parquet for uploads.

---

## App Pages

Navigate using the sidebar. Pipeline status indicators (🟢 / 🔴) show which stages are complete.

### 🏠 Overview

Summary of the project, autoencoder topology diagram, pipeline steps, MPS fault categories, and reported performance metrics.

### 📂 Data Pipeline

**Two modes:**

| Tab | Description |
|---|---|
| ⬆️ Upload Real Data | Upload your own event log (CSV) + MPS signals (CSV or Parquet) |
| 🔬 Generate Synthetic Data | Generate a reproducible synthetic Indus-2 dataset (10 000 events · 50 000 timestamps · 117 supplies) |

After generation, download buttons appear for:
- `indus2_events.csv` — event log
- `indus2_mps.csv` — MPS signals (CSV)
- `indus2_mps.parquet` — MPS signals (Parquet, recommended)

Preview tabs show the event log, beam current trace, and per-supply deviation distributions.

**Expected upload columns:**

*Events CSV:*
```
Time, Ramp end, Kill signal, Event type
```

*MPS CSV / Parquet:*
```
Timestamp, Beam Current, sp1_vmeset, sp1_readback, ..., sp117_vmeset, sp117_readback
```

### 📡 Beam Analysis

Click **▶️ Run Beam State Identification** to:
- Extract beam cycles using the Ramp End → Kill state machine
- Classify each timestamp as No Loss / Partial Loss / Complete Loss
- Visualise cycle timeline, fault overlay on beam current, and cycle duration statistics

### 🧠 Model Training

**Two tabs:**

**Train from Scratch** — configurable epochs and validation split. Training runs epoch-by-epoch with a live progress bar and rolling loss table that update in real time. After training:
- Anomaly threshold is computed from the training set
- Full-dataset inference is run immediately
- Loss curves and model architecture table are shown

**Load Saved Model** — upload a `mps_autoencoder.weights.h5` file. A step-by-step log shows:
1. Architecture rebuild
2. Weight restoration
3. TF graph warm-up (forces JIT compilation on 2 rows)
4. Threshold estimation from a 5 000-row sample
5. Chunked full inference with progress bar

After loading or training, use the **Export Model** tab to download `mps_autoencoder.weights.h5` for reuse.

### 🔍 Anomaly Detection

Shows reconstruction error distribution, anomaly timeline (configurable window), and per-supply fault analysis.

> Supply Fault Analysis runs on-demand — click **▶️ Run Supply Fault Analysis** after setting the slider.

### 📊 Results Dashboard

- Confusion matrix (derived from reported metrics)
- Supply deviation correlation heatmap (configurable number of supplies)
- Per-supply reconstruction error heatmap (configurable supplies × timestamps) — click **▶️ Render Error Heatmap**
- Export tab: download anomaly results as CSV or Parquet

---

## Training on Google Colab

For significantly faster training (T4 GPU), use the included Colab notebook:

1. Open `MPS_Autoencoder_Colab.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set **Runtime → Change runtime type → T4 GPU**
3. Run all cells in order

**What the notebook does:**

| Cell | Action |
|---|---|
| 1 | Install dependencies, verify GPU |
| 2 | Set constants (50 epochs, batch 512) |
| 3 | Generate synthetic Indus-2 dataset |
| 4 | *(Optional)* Mount Google Drive and load real CSVs |
| 5 | Compute deviations |
| 6 | Build autoencoder |
| 7 | Train with EarlyStopping + ReduceLROnPlateau |
| 8 | Plot loss curves |
| 9 | Compute anomaly threshold |
| 10 | Print top-10 highest-error supplies |
| 11 | Save `mps_autoencoder.weights.h5` + `mps_threshold.json` |
| 12 | Download all output files |

After downloading, load the weights in the app via **Model Training → Load Saved Model**.

---

## Model Export & Import

The app uses **weights-only** format (`.weights.h5`) rather than full model serialisation. This avoids Keras version mismatches caused by version-specific config fields (e.g. `quantization_config` present in Keras 3 but absent in earlier versions).

**Export:** Model Training → Export Model tab → *Prepare weights for download*

**Import:** Model Training → Load Saved Model tab → upload `.h5` file

The architecture is always rebuilt from the hardcoded `build_autoencoder()` function before weights are loaded, so the file contains only the weight tensors — no config, no version sensitivity.

---

## Deployment (Streamlit Community Cloud)

Deploy for free at [streamlit.io/cloud](https://streamlit.io/cloud):

1. Push the project to a **public GitHub repository**
2. Go to Streamlit Community Cloud → **New app**
3. Select: repo → branch → `app.py`
4. Click **Deploy**

The app will be live at `https://<username>-<repo-name>.streamlit.app` within ~2 minutes.

> **Note on Streamlit Cloud limits:**
> - RAM: ~1 GB (the autoencoder is ~20 k params, well within this)
> - Upload file size: 200 MB (use Parquet for MPS files, not CSV)
> - CPU only — training 25 epochs takes ~3–5 minutes; use Colab for heavier runs

---

## Technical Reference

### Key Constants (`app.py`)

| Constant | Value | Description |
|---|---|---|
| `NUM_SUPPLIES` | 117 | Number of MPS units |
| `ANOMALY_THRESHOLD_MULTIPLIER` | 2.0 | σ multiplier for threshold |
| `BEAM_PARTIAL_LOSS_THRESHOLD` | 2.0 mA | Drop threshold for Partial Loss |
| `NUM_EVENTS` | 10 000 | Synthetic event log size |
| `NUM_TIMESTAMPS` | 50 000 | Synthetic MPS time series length |

### Session State Keys

| Key | Type | Description |
|---|---|---|
| `event_data` | DataFrame | Event log |
| `mps_data` | DataFrame | Raw MPS signals |
| `deviation_data` | DataFrame | Computed deviations |
| `beam_cycles` | list of tuples | (start, end) pairs |
| `fault_labels` | ndarray | Per-timestamp NL/PL/CL |
| `model` | Keras Model | Trained autoencoder |
| `threshold` | float | Anomaly threshold t_mps |
| `train_history` | dict | Loss curves (None if loaded) |
| `predictions` | ndarray | Binary anomaly flags |
| `recon_errors` | ndarray | Per-timestamp MAE |
| `data_ready` | bool | Pipeline stage flag |
| `beam_ready` | bool | Pipeline stage flag |
| `model_ready` | bool | Pipeline stage flag |

---

*Built with Streamlit · TensorFlow · Plotly · pandas*
