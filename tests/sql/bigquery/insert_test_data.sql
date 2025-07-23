cat > tests/sql/bigquery/insert_test_data.sql << 'EOF'
-- Test Data Insert for Goal Engine
-- Fiscal Fox - Goal Calculation Inputs
-- Master UID: ff_user_8a838f3528819407

-- Insert primary test user data
INSERT INTO `fiscal-fox-fin.fiscal_master_dw.goal_calculation_inputs` (
  master_uid,
  total_debt_balance,
  debt_accounts_json,
  available_monthly_surplus,
  monthly_income_after_tax,
  monthly_fixed_expenses,
  monthly_variable_expenses,
  current_sip_investments,
  risk_tolerance_score,
  calculation_timestamp
) VALUES (
  'ff_user_8a838f3528819407',
  75000.0,
  JSON('[{"balance": 75000, "interest_rate": 18, "lender": "HDFC Credit Card"}]'),
  25000.0,
  80000.0,
  35000.0,
  20000.0,
  5000.0,
  7,
  CURRENT_TIMESTAMP()
);

-- Insert test user with multiple debts
INSERT INTO `fiscal-fox-fin.fiscal_master_dw.goal_calculation_inputs` (
  master_uid,
  total_debt_balance,
  debt_accounts_json,
  available_monthly_surplus,
  monthly_income_after_tax,
  monthly_fixed_expenses,
  monthly_variable_expenses,
  current_sip_investments,
  risk_tolerance_score,
  calculation_timestamp
) VALUES (
  'ff_test_user_multiple_debts',
  125000.0,
  JSON('[
    {"balance": 75000, "interest_rate": 18, "lender": "HDFC Credit Card"},
    {"balance": 50000, "interest_rate": 14, "lender": "SBI Personal Loan"}
  ]'),
  30000.0,
  100000.0,
  40000.0,
  25000.0,
  10000.0,
  8,
  CURRENT_TIMESTAMP()
),
(
  'ff_test_user_low_surplus',
  25000.0,
  JSON('[{"balance": 25000, "interest_rate": 24, "lender": "ICICI Credit Card"}]'),
  5000.0,
  50000.0,
  30000.0,
  15000.0,
  2000.0,
  4,
  CURRENT_TIMESTAMP()
);
EOF
