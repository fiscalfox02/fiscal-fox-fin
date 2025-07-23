#!/bin/bash
set -e

echo " Deploying Fiscal Fox Goal Setting Tables"
PROJECT_ID=${PROJECT_ID:-"pro-kayak-466708-a9"}
MASTER_UID="ff_user_8a838f3528819407"

echo " Creating goal setting tables..."

# Deploy goal calculation inputs
echo "   Creating goal_calculation_inputs..."
envsubst < bigquery/schemas/goal_setting/goal_calculation_inputs.sql | bq query --use_legacy_sql=false

# Deploy enhanced goals table
echo "   Creating user_goals_enhanced..."
envsubst < bigquery/schemas/goal_setting/user_goals_enhanced.sql | bq query --use_legacy_sql=false

echo " Goal setting tables deployed successfully!"
echo " View at: https://console.cloud.google.com/bigquery?project=$PROJECT_ID"
