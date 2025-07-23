cat > tests/sql/validation/verify_goal_calculations.sql << 'EOF'
-- Validation Queries for Goal Engine Testing
-- Run after inserting test data

-- Verify data was inserted correctly
SELECT 
  master_uid,
  total_debt_balance,
  JSON_EXTRACT_ARRAY(debt_accounts_json) as parsed_debts,
  available_monthly_surplus,
  calculation_timestamp
FROM `fiscal-fox-fin.fiscal_master_dw.goal_calculation_inputs`
WHERE master_uid LIKE 'ff_%'
ORDER BY calculation_timestamp DESC;

-- Test JSON parsing functionality
SELECT 
  master_uid,
  JSON_EXTRACT_SCALAR(debt_accounts_json, '$[0].balance') as first_debt_balance,
  JSON_EXTRACT_SCALAR(debt_accounts_json, '$[0].interest_rate') as first_debt_rate,
  JSON_EXTRACT_SCALAR(debt_accounts_json, '$[0].lender') as first_debt_lender
FROM `fiscal-fox-fin.fiscal_master_dw.goal_calculation_inputs`
WHERE master_uid = 'ff_user_8a838f3528819407';

-- Verify goal creation works
SELECT 
  goal_id,
  goal_name,
  goal_type,
  required_monthly_payment,
  required_monthly_sip,
  feasible_with_current_surplus,
  created_at
FROM `fiscal-fox-fin.fiscal_master_dw.user_goals_enhanced`
WHERE master_uid LIKE 'ff_%'
ORDER BY created_at DESC
LIMIT 10;
EOF
