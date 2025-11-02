#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="daring-night-475804-a8"
REGION="asia-southeast2"
REPO="moneylion-202511010"
IMAGE="loan-default-api"

TAG=$(git rev-parse --short HEAD 2>/dev/null || date +%s)

gcloud config set project "$PROJECT_ID"

# gcloud artifacts repositories create $REPO \
    # --repository-format=docker \
    # --location=$REGION \
    # --description="Loan-default prediction containers"


docker buildx build \
  --platform linux/amd64 \
  -t "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}" \
  --push .

# Deploy that specific tag
gcloud run deploy "$IMAGE" \
  --image "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:${TAG}" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1024Mi
