export ETL_RUNNER_SA="etl-runner-sa@daring-night-475804-a8.iam.gserviceaccount.com"

export PROJECT_ID="daring-night-475804-a8"

export BUCKET_NAME="moneylion-202511010"


export COMPOSER_MAIN_SA="etl-runner-sa@daring-night-475804-a8.iam.gserviceaccount.com"

gcloud iam service-accounts add-iam-policy-binding ${ETL_RUNNER_SA} \
  --member="serviceAccount:${COMPOSER_MAIN_SA}" \
  --role="roles/iam.serviceAccountUser"
