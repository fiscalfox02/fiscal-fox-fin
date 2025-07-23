#!/bin/bash
set -e

echo " Deploying Goal Engine with BigQuery Integration..."

# Deploy create goal endpoint
gcloud functions deploy goal-engine-create \
  --runtime python39 \
  --trigger-http \
  --entry-point create_goal_endpoint \
  --source . \
  --allow-unauthenticated \
  --project fiscal-fox-fin \
  --region us-central1 \
  --memory 512MB \
  --timeout 60s

# Deploy get goals endpoint
gcloud functions deploy goal-engine-get \
  --runtime python39 \
  --trigger-http \
  --entry-point get_goals_endpoint \
  --source . \
  --allow-unauthenticated \
  --project fiscal-fox-fin \
  --region us-central1 \
  --memory 256MB \
  --timeout 30s

echo "Goal Engine deployed successfully!"
echo "Create Goal: https://us-central1-fiscal-fox-fin.cloudfunctions.net/goal-engine-create"
echo "Get Goals: https://us-central1-fiscal-fox-fin.cloudfunctions.net/goal-engine-get"
