cat > scripts/testing/run_goal_tests.sh << 'EOF'
#!/bin/bash
set -e

echo " Running Complete Goal Engine Test Suite"

# Step 1: Populate test data
echo "Step 1: Populating test data..."
./scripts/database/populate_test_data.sh

# Step 2: Test timeline goal endpoint
echo ""
echo "Step 2: Testing timeline goal creation..."
curl -X POST https://us-central1-fiscal-fox-fin.cloudfunctions.net/goal-engine-create \
  -H "Content-Type: application/json" \
  -d '{
    "master_uid": "ff_user_8a838f3528819407",
    "goal_type": "timeline",
    "goal_name": "Test: Clear Credit Card Debt",
    "target_date": "2026-12-31"
  }' | jq '.'

echo ""

# Step 3: Test investment goal endpoint
echo "Step 3: Testing investment goal creation..."
curl -X POST https://us-central1-fiscal-fox-fin.cloudfunctions.net/goal-engine-create \
  -H "Content-Type: application/json" \
  -d '{
    "master_uid": "ff_user_8a838f3528819407",
    "goal_type": "investment",
    "goal_name": "Test: House Down Payment",
    "target_amount": 500000,
    "horizon_years": 3
  }' | jq '.'

echo ""

# Step 4: Test get goals endpoint
echo "Step 4: Testing get goals endpoint..."
curl "https://us-central1-fiscal-fox-fin.cloudfunctions.net/goal-engine-get?master_uid=ff_user_8a838f3528819407" | jq '.'

echo ""
echo " Goal Engine test suite completed!"
echo "Run 'bq query --use_legacy_sql=false < tests/sql/bigquery/cleanup_test_data.sql' to clean up"
EOF

chmod +x scripts/testing/run_goal_tests.sh
