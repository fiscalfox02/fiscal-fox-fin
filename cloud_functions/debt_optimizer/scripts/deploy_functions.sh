#!/bin/bash

set -e

PROJECT_ID=${GCP_PROJECT_ID:-"fiscal-fox-fin"}
REGION=${GCP_REGION:-"us-central1"}

echo " Deploying Debt Optimizer Cloud Functions..."

# Deploy debt analyzer function
echo " Deploying debt-analyzer-api function..."
gcloud functions deploy debt-analyzer-api \
    --runtime python39 \
    --trigger-http \
    --allow-unauthenticated \
    --memory 1GB \
    --timeout 540s \
    --source src/api \
    --entry-point debt_analyzer_api \
    --region $REGION \
    --set-env-vars GCP_PROJECT_ID=$PROJECT_ID,ENV=production

echo " Debt Optimizer Cloud Functions deployed successfully!"
