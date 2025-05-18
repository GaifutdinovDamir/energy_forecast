# Energy Forecast Project

## Overview
This project predicts electricity usage for the next 24 hours using time series data and CatBoost model.  
It includes data preprocessing, feature engineering, validation, model training, and saving artifacts. It is done as a training task for developing my ml system design skills.

---

## Setup

### Requirements
- Python 3.12+
- Poetry (for dependency management)

### Installation

1. Clone the repo:
   ```bash
   git clone git@github.com:GaifutdinovDamir/energy_forecast.git
   cd energy_forecast

2.	Install dependencies with Poetry:
    ```bash
    install poetry
## Usage

1. Data Preprocessing
Preprocess all raw CSV files from data/raw_data/ and save processed data to data/processed_data/.
    ```bash
    PYTHONPATH=. poetry run python src/data/preprocess_data.py
2. Data Validation
Run validation on all processed CSV files to ensure data quality:
    ```bash
    PYTHONPATH=. poetry run python src/data/validate_data.py
3. Run Full Pipeline
Run the full pipeline: load processed data, generate features, train CatBoost model, and save artifacts.
    ```bash
    PYTHONPATH=. poetry run python src/pipeline/run_pipeline.py
## Project Structure
```
├── artifacts
├── data
│   ├── processed_data
│   │   ├── AEP_hourly.csv
│   │   ├── COMED_hourly.csv
│   │   └── …
│   └── raw_data
│       ├── AEP_hourly.csv
│       ├── COMED_hourly.csv
│       └── …
├── logs
│   ├── log_preprocess_data_YYYYMMDD_HHMMSS.log
│   └── …
├── poetry.lock
├── pyproject.toml
└── src
├── data
│   ├── preprocess_data.py
│   └── validate_data.py
├── features
│   └── make_features.py
├── model
│   └── train_model.py
├── pipeline
│   ├── conf.json
│   └── run_pipeline.py
└── utils
├── data_utils.py
├── data_validation.py
└── logging_utils.py