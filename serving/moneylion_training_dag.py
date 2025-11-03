from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

# block handles different versions of the Kubernetes provider,
try:
    from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
except ModuleNotFoundError:
    from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

GCP_PROJECT_ID = "daring-night-475804-a8"
GCP_REGION     = "asia-southeast2"
GCP_REPO       = "moneylion-202511010"
IMAGE_NAME     = "loan-default-api"
IMAGE_URI      = f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/{GCP_REPO}/{IMAGE_NAME}:latest"


KUBE_CONN_ID = "kubernetes_default"
NAMESPACE    = "composer-user-workloads"


with DAG(
    dag_id="moneylion_training_pipeline",
    start_date=pendulum.datetime(2025, 11, 1, tz="UTC"),
    schedule=None,
    catchup=False,
    tags=["moneylion", "ml"],
) as dag:

    run_full_pipeline = KubernetesPodOperator(
        task_id="run_full_pipeline",
        name="pod-full-pipeline",
        namespace=NAMESPACE,
        image=IMAGE_URI,
        kubernetes_conn_id=KUBE_CONN_ID,
        service_account_name="default",
        is_delete_operator_pod=True,
        get_logs=True,
        log_events_on_failure=True,
        image_pull_policy="Always",
        cmds=["python", "main.py"],
        startup_timeout_seconds=1800,  
    )

    rollout_cloud_run = BashOperator(
        task_id="rollout_cloud_run",
        bash_command=(
            "gcloud run services update loan-default-api "
            f"--region={GCP_REGION} --platform=managed "
            "--set-env-vars="
            "RELOAD_TIMESTAMP={{ ts_nodash }},"
            "GCS_ARTIFACT_BUCKET=moneylion-202511010,"
            "GCS_ARTIFACT_PREFIX=artifacts"
        ),
    )

    run_full_pipeline >> rollout_cloud_run

