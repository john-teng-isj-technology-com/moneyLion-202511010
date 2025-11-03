# moneyLion-202511010

MoneyLion Loan Repayment Prediction Pipeline

This repository contains an end-to-end machine learning pipeline for predicting loan repayment defaults (isBadDebt) at MoneyLion. It processes raw loan data through ingestion, transformation, preprocessing, embedding, and XGBoost training, then deploys a prediction API to Google Cloud Run. The system is orchestrated via Apache Airflow and uses MLflow for experiment tracking.
The project was developed as part of a Data Scientist assessment, demonstrating modular MLOps practices, reproducible workflows, and business impact analysis. For more details on the assessment guidelines, see DS Take Home Guidelines.pdf.
Features

Modular Pipeline: Stages for data ingestion, transformation (z-scoring, one-hot encoding), preprocessing (splits, vocabularies), embedding (learned dense vectors), and XGBoost training.
MLOps Integration: Airflow DAG for orchestration, MLflow for logging metrics/artifacts, GCS for artifact storage.
Inference API: Flask-based service with /predict endpoint for batch predictions and a simple web form for single predictions.
Analysis Tools: Scripts to generate predictions, compute financial impacts, create charts, and produce architecture diagrams.
Deployment: Cloud Run service with automatic artifact download on startup; supports rollout without image rebuilds.
Impact Assessment: Calculates business metrics like profit uplift (e.g., from 10.19% to 39.66%) and bad debt reduction.

Live Demo: Loan Default Prediction API
Technology Stack

Core Libraries: Pandas, NumPy, Scikit-learn, XGBoost, PyTorch (for embeddings)
MLOps: MLflow (experiment tracking), Apache Airflow (orchestration)
Deployment: Google Cloud Run, Artifact Registry, Cloud Storage, Cloud Build
Serving: Flask, Gunicorn
Utilities: Graphviz (diagrams), Seaborn/Matplotlib (charts)
Environment: Python 3.12, Docker

Installation


Clone the repository:
git clone https://github.com/john-teng-isj-technology-com/moneyLion-202511010.git
cd moneyLion-202511010



Set up a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate



Install dependencies:
pip install -r requirements.txt



Configure Google Cloud credentials (for GCS access):

Set GOOGLE_APPLICATION_CREDENTIALS environment variable to your service account key path.
Or use Application Default Credentials (ADC) via gcloud auth application-default login.



(Optional) For diagrams: Install Graphviz system binaries (see troubleshooting if needed).


Usage
Running the Training Pipeline Locally


Ensure raw data is available in GCS or locally.


Run the main pipeline:
python main.py

This executes all stages: ingestion → transformation → preprocessing → embedding → training.
Artifacts are saved in artifacts/ and uploaded to GCS.


View MLflow runs:
mlflow ui --port 5000

Open http://localhost:5000 in your browser.


Running Analysis Scripts
After training, generate predictions and impacts:
python generate_predictions.py
python analyse_metrics.py
python generate_charts.py
python create_diagrams.py

Outputs appear in artifacts/analysis/.
Deploying the API


Build and push the Docker image (configure Cloud Build or use bash/cloudBuild.sh).


Deploy to Cloud Run via the Airflow DAG or manually:
gcloud run deploy loan-default-api \
  --image=asia-southeast2-docker.pkg.dev/your-project/moneylion-202511010/loan-default-api:latest \
  --region=asia-southeast2 \
  --platform=managed \
  --allow-unauthenticated



Test the API:

Health: curl https://your-service-url/health
Predict: curl -X POST https://your-service-url/predict -H "Content-Type: application/json" -d '{"rows": [{"apr": 0.3, "loanAmount": 1000, ...}]}'



Orchestration with Airflow

Copy serving/moneylion_training_dag.py to your Airflow DAGs folder.
Trigger the DAG in Composer to run training and rollout.

Pipeline Overview
The pipeline consists of five modular stages:

Data Ingestion: Downloads raw CSV from GCS to local artifacts.
Data Transformation: Joins datasets, applies z-scoring, one-hot encoding; saves stats and dummy columns.
Data Preprocessing: Builds vocabularies, creates categorical indices, splits data; saves npy arrays and metadata.
Embedding & Export: Trains embeddings for categoricals, exports dense matrices (X_*) and schema.
Model Training: Grid search on XGBoost, evaluates on val/test, logs to MLflow, saves model and reports.

All stages upload artifacts to GCS. For diagrams, run create_diagrams.py to generate Graphviz DOT and Mermaid files (import to draw.io).
Deployment

Image Building: Use Cloud Build (cloudBuild.yaml) or local Docker with bash/cloudBuild.sh.
Cloud Run: Deploys the Flask app; on startup, downloads artifacts if missing.
Orchestration: Single-pod Airflow DAG runs the full pipeline and triggers rollout.

For more on MoneyLion's actual products, see MoneyLion: Banking & Rewards or MoneyLion on GitHub.
Contributing

Fork the repo.
Create a feature branch (git checkout -b feature/AmazingFeature).
Commit changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/AmazingFeature).
Open a Pull Request.

Please follow Python best practices and add tests for new features.