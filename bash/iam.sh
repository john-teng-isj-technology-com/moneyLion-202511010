PROJECT_ID=daring-night-475804-a8
REGION=asia-southeast2
REPO=loan-default-models
SA=etl-runner-sa@daring-night-475804-a8.iam.gserviceaccount.com
USER_EMAIL="johnkervalk@gmail.com"


gcloud artifacts repositories add-iam-policy-binding $REPO \
  --location=$REGION \
  --member=serviceAccount:$SA \
  --role=roles/artifactregistry.writer

gcloud iam service-accounts add-iam-policy-binding $SA \
  --member=user:$(gcloud config get-value account) \
  --role=roles/iam.serviceAccountUser
