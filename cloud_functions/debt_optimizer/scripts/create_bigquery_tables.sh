#!/bin/bash

set -e

PROJECT_ID=${GCP_PROJECT_ID:-"fiscal-fox-fin"}
DATASET_ID="debt_analytics"

echo " Creating BigQuery dataset and tables..."

# Create dataset
bq mk --dataset \
    --description "Debt Optimization Analytics Dataset" \
    --location=US \
    $PROJECT_ID:$DATASET_ID || echo " Dataset already exists"

# Create tables
TABLES=("debt_portfolio" "debt_optimization_results" "debt_webhook_cache")

for table in "${TABLES[@]}"; do
    echo " Creating $table table..."
    bq mk --table \
        $PROJECT_ID:$DATASET_ID.$table \
        schemas/${table}_schema.json || echo "ℹTable $table already exists"
done

echo " BigQuery tables created successfully!"
