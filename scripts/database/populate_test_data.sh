cat > scripts/database/populate_test_data.sh << 'EOF'
#!/bin/bash
set -e

echo " Populating Goal Engine Test Data"

PROJECT_ID=${PROJECT_ID:-"fiscal-fox-fin"}
DATASET_ID="fiscal_master_dw"

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
    echo " Not authenticated with Google Cloud"
    echo "Run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project $PROJECT_ID

echo " Creating tables if they don't exist..."
bq query --use_legacy_sql=false < tests/sql/bigquery/create_goal_tables.sql

echo "Inserting test data into goal_calculation_inputs..."
bq query --use_legacy_sql=false < tests/sql/bigquery/insert_test_data.sql

echo "Test data inserted successfully"

echo "Validating data insertion..."
bq query --use_legacy_sql=false < tests/sql/validation/verify_goal_calculations.sql

echo " Validation complete"

echo "Test data summary:"
echo "   - Primary test user: ff_user_8a838f3528819407"
echo "   - Multiple debts user: ff_test_user_multiple_debts" 
echo "   - Low surplus user: ff_test_user_low_surplus"

echo " Ready to test Goal Engine endpoints!"
EOF

chmod +x scripts/database/populate_test_data.sh
