## 🧠 Demand Forecasting & Inventory Optimization System

End-to-end Machine Learning system designed to forecast product demand and optimize inventory decisions using uncertainty-aware predictions.

---

## Project Overview

This project simulates a **production-grade ML system** used in retail and e-commerce to:

* Forecast future product demand
* Estimate prediction uncertainty
* Recommend optimal stock levels
* Reduce stockouts and overstock
* Provide actionable business decisions via API

---

## System Architecture

```
Raw Data → Preprocessing → Feature Engineering → Model Training
        → Evaluation → Model Registry → Inference Pipeline → API
```

---

## 📂 Project Structure

### 🔹 `api/`

**Role:** API layer (FastAPI)

* `main.py`

  * Entry point of the application
  * Exposes `/predict` and `/train` endpoint
  * Handles requests and responses
  * Calls inference pipeline

---

### 🔹 `config/`

**Role:** Configuration management

* `config.yaml`

  * Model parameters
  * Paths
  * Training settings
  * Easily modifiable without touching code

---

### 🔹 `data/`

#### `data/raw/`

* Original dataset (CSV)
* Untouched source data

#### `data/processed/`

* `train.csv` / `test.csv`
* Cleaned and split datasets ready for modeling

---

### 🔹 `notebooks/`

**Role:** Exploration & experimentation

* `eda.ipynb`

  * Data analysis and exploration
---

### 🔹 `src/` (Core ML Logic)

---

#### 📁 `data/`

* `preprocessing.py`

  * Data cleaning
  * Handling missing values
  * Formatting dates
  * Preparing dataset for feature engineering

---

#### 📁 `features/`

* `build_features.py`

  * Creation of time-series features:

    * Lag features
    * Rolling averages
    * Time-based features
  * Transforms raw data into model-ready features

---

#### 📁 `models/`

* `train.py`

  * Model training pipeline
  * Uses XGBoost
  * Handles:

    * Training
    * Validation
    * Saving model artifacts

---

#### 📁 `evaluation/`

* `metrics.py`

  * Computes evaluation metrics:

    * RMSE
    * MAE
    * MAPE

* `offline_metrics.py`

  * Model evaluation after training
  * Generates performance reports
  * Produces plots for analysis

---

#### 📁 `pipeline/`

* `inference.py`

  * Core prediction logic
  * Applies feature engineering
  * Generates predictions
  * Computes uncertainty
  * Calculates business decisions

👉 Key outputs:

* `predicted_sales`
* `prediction_lower / upper`
* `safety_stock`
* `recommended_stock`
* `decision`

---

### 🔹 `utils/`

* `paths.py`

  * Centralized path management

* `config.py`

  * Loads and parses configuration file

* `utils.py`

  * Helper functions used across the project

---

##  Model

* Algorithm: **XGBoost**
* Features:

  * Lag variables
  * Rolling statistics
  * Time-based features

---

## Inference Logic

The system does more than prediction:

```python
recommended_stock = predicted_sales + safety_stock
```

Where:

* `safety_stock = Z * RMSE`
* Z = service level (1.65 ≈ 95%)

---
Exact 👍 tu as raison — il faut corriger cette section pour refléter **les deux endpoints : `/train` + `/predict`**.

Voici la **version propre et corrigée** que tu peux remplacer directement 👇

---

## 🚀 API Usage

---

### 🔹 1. Train Model

#### Endpoint

```
POST /train
```
#### Description

Trains the model using a CSV dataset.

#### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/train" \
  -F "file=@data.csv"
```

#### Example Response

```json
{
  "rmse": 12.45,
  "mae": 8.32,
  "model_path": "models/model.pkl"
}
```
---

### 🔹 2. Predict
#### Endpoint

```
POST /predict
```

#### Description

Generates demand forecasts and inventory recommendations using the trained model.

#### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@data.csv"
```

#### Example Response

```json
{
  "summary": {
    "total_predicted_sales": 1606063.0,
    "total_recommended_stock": 1678740.66,
    "average_uncertainty": 6.27
  },
  "predictions": [...]
}
```
---

## Business Value

This system enables:

* Better inventory planning
* Reduced stockouts
* Lower overstock costs
* Data-driven decision making


## Future Improvements

* Add model retraining pipeline
* Deploy to cloud (GCP / AWS)
* Add monitoring & drift detection
* Build lightweight frontend dashboard

---

#  Installation & Run

## Clone the repository

```bash
git clone https://github.com/Bayon-idris/smart-demand-forecasting-ml-system.git
cd smart-demand-forecasting-ml-system
```

---

## 🐍 Create virtual environment

```bash
python -m venv venv
```

### Activate it

**Windows**

```bash
venv\Scripts\activate
```
---

## 📦 Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn api.main:app --reload
```
## ▶️ Run the train model

python.exe -m src.models.train

---

