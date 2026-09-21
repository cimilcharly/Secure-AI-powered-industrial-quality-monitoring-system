# Secure AI-Powered Industrial Quality Monitoring System — Fabric Defect Inspection

[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io/)
[![Security](https://img.shields.io/badge/Security-AES--256--GCM%20%7C%20SHA--256-blue)](#security-layer)
[![ML](https://img.shields.io/badge/Engine-YOLOv8m%20%7C%20PatchCore-purple)](#ai-engines)
[![Federated](https://img.shields.io/badge/Federated-FedAvg%20(3--Node)-orange)](#federated-learning-3-node-simulation)

An industrial automated fabric defect inspection and anomaly detection platform built on a manually collected dataset. It detects known fabric defects across **4 confirmed categories**, flags previously unseen anomalies, scores severity and estimated cost impact in **Indian Rupees (₹)**, explains its predictions visually with **Grad-CAM**, monitors performance **drift**, secures all data in transit and at rest, and includes a working **3-node federated learning demonstration across 3 simulated factories**.

---

## 1. Project Architecture

```
Camera / Uploaded Image
        │
   AES-256-GCM Encryption (in transit)
        │
   FastAPI Backend  <── JWT Auth & Role Check (Admin / Operator)
        │
   SHA-256 Tamper & Integrity Check
        │
   Preprocessing (CLAHE contrast, Bilateral denoise, Standardization)
        │
   ───────────────────────────────
   │                             │
YOLOv8m Defect Detection   PatchCore Anomaly Detection
(Damage, Button, Stitch, Color) (k-NN Nominal Memory Bank)
   │                             │
   ───────────────────────────────
        │
   Severity Scoring (0–100 Scale) & Cost Estimation (₹ Rework/Scrap/Recall)
        │
   Grad-CAM Explainability (Visual Heatmap + Localization Recall Metric)
        │
   Embedding Drift Monitoring (Cosine distance to baseline)
        │
   Database Persistence (MySQL with SQLite local fallback, tagged by factory_id)
        │
   Streamlit Monitoring Dashboard (with multi-factory filter)
        │
   Federated Learning Sync Layer (3-Factory FedAvg with AES-encrypted weights)
```

---

## 2. Confirmed Defect Taxonomy (4 YOLO Classes)

| Category (YOLO class) | Owner | Sub-variations Captured |
|---|---|---|
| **Damage** (0) | Barath | Tear, hole, cut, burn mark, frayed edge, scratch, fabric rip |
| **Button** (1) | Dhakshina | Missing, loose, wrong color, wrong size, broken, misaligned |
| **Stitch** (2) | Cimil | Broken stitch, missing, double stitch, loose thread, open seam |
| **Color** (3) | Kevin | Color fade, mismatch, dye patch, dark spot, ink stain, shade variation |

---

## 3. Team Work Distribution

| Member | Primary (Data Ownership) | Secondary (Pipeline Modules) |
|---|---|---|
| **Barath** | Damage Category data, annotation, synthetic generation | YOLOv8m detection training/eval (Mod 4), Federated Learning 3-node extension (Mod 13) |
| **Dhakshina** | Button Category data, annotation, synthetic generation | Severity + ₹ Cost Estimation (Mod 6), Grad-CAM Explainability (Mod 7), Streamlit UI (Mod 12) |
| **Cimil** | Stitch Category data, annotation, synthetic generation | Preprocessing (Mod 3), PatchCore Anomaly (Mod 5), Drift Detection (Mod 8) |
| **Kevin** | Color Category data, annotation, synthetic generation | Security Layer (Mod 9), FastAPI Backend (Mod 10), MySQL Database (Mod 11), Deployment (Mod 14) |

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- MySQL (Optional: automatically falls back to local SQLite `fabric_qc.db` if MySQL is not running)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated Smoke Test (All 14 Modules)
Verify every pipeline component end-to-end:
```bash
python smoke_test.py
```

### Step 3: Launch FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### Step 4: Launch Streamlit Dashboard
Open a new terminal:
```bash
streamlit run dashboard/app.py
```
- Local URL: [http://localhost:8501](http://localhost:8501)

---

## 5. Security & Default Credentials

- **Admin Account**:
  - Username: `admin`
  - Password: `admin123`
  - Permissions: Full inspection, retrain triggers, ₹ cost matrix tuning, user management
- **Operator Account**:
  - Username: `operator`
  - Password: `operator123`
  - Permissions: Real-time inspection feeds, historical logs, drift viewing
- **AES-256-GCM**: Automatic authenticated symmetric payload encryption.
- **SHA-256**: Cryptographic digest computed prior to transmission and verified upon receipt.

---

## 6. Training Pipelines (Weeks 12–13)

### YOLOv8m Defect Detection Training:
```bash
python src/detection/train_yolo.py --data data/dataset.yaml --epochs 100 --batch 16 --device 0
```

### PatchCore Nominal Memory Bank Training:
```bash
python src/anomaly_detection/train_patchcore.py --data data/raw/defect_free --output models/patchcore_memory_bank.npy
```

### 3-Node Federated Learning Simulation (CLI):
```bash
python -c "from src.federated.fedavg_simulation import run_federated_simulation; print(run_federated_simulation(rounds=5))"
```

---

## 7. Docker Deployment

To launch MySQL, FastAPI backend, and Streamlit dashboard orchestrated together:
```bash
docker-compose -f docker/docker-compose.yml up --build
```
