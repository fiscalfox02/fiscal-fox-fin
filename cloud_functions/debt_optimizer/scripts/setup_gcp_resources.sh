#!/bin/bash

set -e

PROJECT_ID=${GCP_PROJECT_ID:-"fiscal-fox-fin"}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_ACCOUNT_NAME="debt-optimizer-sa"

echo "Setting up GCP resources for Debt Optimizer..."

# Set project
gcloud config set project $PROJECT_ID

# Enable APIs
echo " Enabling required APIs..."
gcloud services enable bigquery.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable container.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Create service account
echo " Creating service account..."
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
    --display-name="Debt Optimizer Service Account" \
    --description="Service account for debt optimization system" || true

# Grant permissions
echo " Granting permissions..."
ROLES=(
    "roles/bigquery.dataEditor"
    "roles/bigquery.user"
    "roles/cloudsql.client"
    "roles/storage.objectViewer"
)

for role in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="$role"
done

# Create service account key
if [ ! -f "service-account-key.json" ]; then
    echo " Creating service account key..."
    gcloud iam service-accounts keys create service-account-key.json \
        --iam-account=$SERVICE_ACCOUNT_NAME@$PROJECT_ID.iam.gserviceaccount.com
    echo " Remember to add this key to GitHub Secrets as GCP_SA_KEY"
fi

echo "GCP resources setup completed!"
