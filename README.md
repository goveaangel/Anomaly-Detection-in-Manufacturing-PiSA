# Anomaly Detection in Manufacturing - PiSA Farmacéutica

This repository presents an **Exploratory Data Analysis (EDA)** and **Anomaly Detection** project for PiSA Farmacéutica. The study focuses on detecting critical anomalies in temperature variables recorded from a blow molding machine used in the production of Electrolit serum bottles. These findings aim to support **predictive maintenance strategies** and extend the operational life of the equipment.

## 📝 Project Overview

The blow molding machine at PiSA Farmacéutica has experienced premature wear and failure, operating only **2-3 months** before requiring major maintenance, far below the expected **1-year** lifespan. Through this project, we analyzed temperature-related data from over **100 sensors**, applied anomaly detection algorithms, and validated our findings using **SARIMA time series forecasting models**.

The analysis confirms that **thermal stress**, as evidenced by anomalous fluctuations in key temperature variables, contributes to the accelerated deterioration of the machine.

## 📊 Main Objectives

- Perform **Exploratory Data Analysis (EDA)** on large-scale industrial datasets.
- Apply **Anomaly Detection** using K-Nearest Neighbors (KNN) and PyOD.
- Validate anomalies and predict variable behavior with **SARIMA** forecasting models.
- Provide actionable insights for **predictive maintenance**.

---

## 📁 Repository Structure

At the root of the repository, you will find the project report files:

```
├── Report_PiSA.pdf            # Detailed methodology and results (English)
├── Reporte_PiSA.pdf           # Detailed methodology and results (Spanish)
```

The implementation code is organized into two main folders:

```
├── notebooks/                 # Jupyter Notebooks for anomaly detection and forecasting
│   ├── sarimax_var1.ipynb           # SARIMA model for variable 1
│   ├── sarimax_var2.ipynb           # SARIMA model for variable 2
│   ├── sarimax_var3.ipynb           # SARIMA model for variable 3
│   └── variables_temp_prueba_pyod.ipynb  # Anomaly detection with KNN (PyOD library)

├── scripts/                   # Python scripts for data processing and preparation
│   ├── main1.py                    # Data preprocessing and EDA script
│   └── main_semana.py              # Time-windowed data analysis and preparation
```

---

## 🔧 Tools & Technologies

- **Python 3.10+**
- **Pandas**, **NumPy**, **PyArrow** (for data handling and processing)
- **PyOD** (for anomaly detection)
- **Statsmodels SARIMAX** (for time series forecasting)
- **Matplotlib**, **Seaborn** (for data visualization)

---

## 🔬 Methodology

### 1. Data Collection & Preprocessing
- Data sourced from over 100 sensors.
- Files analyzed: **November 2024** and **January 2025** datasets (~1.8GB each).
- Data structured in Parquet format; processed in **chunks** of 100,000 records to manage memory.
- Pivoted data for easier time series analysis and selected the top **temperature-related** variables based on data coverage.

### 2. Anomaly Detection
- **K-Nearest Neighbors (KNN)** algorithm implemented using **PyOD**.
- Time series smoothed with a **10-observation moving average**.
- Parameters:
  - `n_neighbors = 20`
  - `contamination = 0.0025` (0.25% of data assumed anomalous)

### 3. Forecasting & Validation
- **SARIMA** models applied to three key variables:
  - `energyPerPreform_CurrentPreformNeckFinishTemperature.0`
  - `powerPerPreform_CurrentPreformNeckFinishTemperature.0`
  - `numberOfActivatedRadiators_CurrentPreformNeckFinishTemperature.0`
- Sampling interval adjusted to **10 seconds** to reduce computational load.
- Model validated with **ADF**, **ACF**, **PACF** tests.
- Evaluation metrics: **MAE** and **RMSE**.

---

## 📈 Results

- Identified critical anomalies where **multiple temperature variables** showed deviations **simultaneously**.
- **SARIMA** forecasts validated the anomalies, confirming significant deviations from expected behavior.
- Results highlight the impact of **thermal stress** on machine lifespan and support the hypothesis that **predictive maintenance** can reduce downtime.

---

## 🏭 Business Impact

This project provides:
- A framework for **real-time anomaly detection**.
- A **predictive maintenance** strategy to minimize unscheduled downtime.
- A foundation for **improving operational efficiency** and **reducing maintenance costs** at PiSA Farmacéutica.

---

## 📜 Report

For a detailed explanation of the problem, methodology, and results, please refer to the **[Report_PiSA.pdf](./Report_PiSA.pdf)** included in this repository.

---

## 📫 Contact

For questions or collaboration opportunities, feel free to reach out:

- **José Ángel Govea García** - [goveaangel090@gmail.com](mailto:goveaangel090@gmail.com)
