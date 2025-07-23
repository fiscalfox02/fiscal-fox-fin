#!/bin/bash
set -e

echo "🎯 Deploying Fiscal Fox Goal Engine..."

PROJECT_ID=${PROJECT_ID:-"fiscal-fox-fin"}

echo "📊 Deploying Goal Engine Cloud Functions..."

# Deploy create goal endpoint
gcloud functions deploy goal-engine-create \
  --runtime python39 \
  --trigger-http \
  --entry-point create_goal \
  --source . \
  --allow-unauthenticated \
  --project $PROJECT_ID \
  --region us-central1 \
  --memory 512MB \
  --timeout 60s

# Deploy get goals endpoint
gcloud functions deploy goal-engine-get \
  --runtime python39 \
  --trigger-http \
  --entry-point get_goals \
  --source . \
  --allow-unauthenticated \
  --project $PROJECT_ID \
  --region us-central1 \
  --memory 256MB \
  --timeout 30s

echo "✅ Fiscal Fox Goal Engine deployed successfully!"
echo "🎯 Create Goal: https://us-central1-fiscal-fox-fin.cloudfunctions.net/goal-engine-create"
echo "📊 Get Goals: https://us-central1-fiscal-fox-fin.cloudfunctions.net/goal-engine-get"
echo "📊 View at: https://console.cloud.google.com/functions/list?project=$PROJECT_ID"
