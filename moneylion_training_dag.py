from __future__ import annotations

import pendulum

from airflow.models.dag import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator

GCP_PROJECT_ID = "daring-night-475804-a8"
GCP_REGION = "asia-southeast2"
GCP_REPO = "moneylion-202511010"
IMAGE_NAME = "loan-default-api"
IMAGE_URI = f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/{GCP_REPO}/{IMAGE_NAME}:latest"

COMPOSER_SERVICE_ACCOUNT = "etl-runner-sa@daring-night-475804-a8.iam.gserviceaccount.com"

with DAG(
    dag_id="moneylion_training_pipeline",
    start_date=pendulum.datetime(2025, 11, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["moneylion", "ml"],
) as dag:

    data_transformation = KubernetesPodOperator(
        task_id="data_transformation",
        name="pod-data-transformation",
        namespace="default",
        image=IMAGE_URI,
        cmds=["python", "-m", "src.moneylion.pipeline.data_transformation_pipeline"],
        service_account_name="default", 
        do_xcom_push=False,
        get_logs=True,
        log_events_on_failure=True,
        in_cluster=True,
        startup_timeout_seconds=300,
    )

    data_preprocessing = KubernetesPodOperator(
        task_id="data_preprocessing",
        name="pod-data-preprocessing",
        namespace="default",
        image=IMAGE_URI,
        cmds=["python", "-m", "src.moneylion.pipeline.data_preprocessing_pipeline"],
        service_account_name="default",
        do_xcom_push=False,
        get_logs=True,
        log_events_on_failure=True,
        in_cluster=True,
        startup_timeout_seconds=300,
    )

    data_embedding = KubernetesPodOperator(
        task_id="data_embedding",
        name="pod-data-embedding",
        namespace="default",
        image=IMAGE_URI,
        cmds=["python", "-m", "src.moneylion.pipeline.embedding_pipeline"],
        service_account_name="default",
        do_xcom_push=False,
        get_logs=True,
        log_events_on_failure=True,
        in_cluster=True,
        startup_timeout_seconds=600, 
    )

    model_training = KubernetesPodOperator(
        task_id="model_training",
        name="pod-model-training",
        namespace="default",
        image=IMAGE_URI,
        cmds=["python", "-m", "src.moneylion.pipeline.model_training_pipeline"],
        service_account_name="default",
        do_xcom_push=False,
        get_logs=True,
        log_events_on_failure=True,
        in_cluster=True,
        startup_timeout_seconds=600, 
    )

    data_transformation >> data_preprocessing >> data_embedding >> model_training

