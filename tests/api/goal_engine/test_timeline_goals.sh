cat > tests/api/goal_engine/test_timeline_goals.sh << 'EOF'
#!/bin/bash

echo "Testing Timeline Goal API Endpoints"

BASE_URL="https://us-central1-fiscal-fox-fin.cloudfunctions.net"

# Test 1: Basic timeline goal
echo "Test 1: Basic timeline goal creation"
curl -X POST $BASE_URL/goal-engine-create \
  -H "Content-Type: application/json" \
  -d '{
    "master_uid": "ff_user_8a838f3528819407",
    "goal_type": "timeline",
    "goal_name": "Clear Credit Card Debt",
    "target_date": "2026-12-31"
  }' | jq '.'

echo ""

# Test 2: Timeline goal with custom debts
echo "Test 2: Timeline goal with custom debts"
curl -X POST $BASE_URL/goal-engine-create \
  -H "Content-Type: application/json" \
  -d '{
    "master_uid": "ff_user_8a838f3528819407",
    "goal_type": "timeline",
    "goal_name": "Clear All Loans",
    "target_date": "2027-06-30",
    "debts": [
      {"balance": 50000, "rateOfInterest": 15},
      {"balance": 25000, "rateOfInterest": 20}
    ]
  }' | jq '.'

echo ""
echo "Timeline goal tests completed"
EOF

chmod +x tests/api/goal_engine/test_timeline_goals.sh
