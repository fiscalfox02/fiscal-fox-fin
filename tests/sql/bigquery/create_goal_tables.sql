cat > tests/sql/bigquery/create_goal_tables.sql << 'EOF'
-- Create Goal Setting Tables for Fiscal Fox
-- Based on your existing schemas

-- Set project variable
DECLARE PROJECT_ID STRING DEFAULT 'fiscal-fox-fin';

-- Create goal_calculation_inputs table
CREATE TABLE IF NOT EXISTS `fiscal-fox-fin.fiscal_master_dw.goal_calculation_inputs` (
  master_uid STRING NOT NULL,
  calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  
  -- Debt inputs for timeline goals
  total_debt_balance FLOAT64,
  weighted_avg_interest_rate FLOAT64,
  debt_accounts_json JSON,
  
  -- Investment inputs for recurring/investment goals
  total_liquid_savings FLOAT64,
  current_sip_investments FLOAT64,
  existing_portfolio_breakdown JSON,
  
  -- Income and expense inputs for feasibility
  monthly_income_after_tax FLOAT64,
  monthly_fixed_expenses FLOAT64,
  monthly_variable_expenses FLOAT64,
  current_emi_obligations FLOAT64,
  
  -- Goal setting parameters
  available_monthly_surplus FLOAT64,
  surplus_confidence_level FLOAT64,
  risk_tolerance_score INT64,
  investment_time_horizon_years INT64,
  
  -- Emergency fund calculations
  emergency_fund_target_months INT64,
  current_emergency_fund FLOAT64,
  
  -- Micro goal inputs
  paydays_per_month INT64,
  discretionary_spending_weekly FLOAT64,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(created_at)
CLUSTER BY master_uid
OPTIONS(
  description="Goal calculation inputs for Fiscal Fox with amortization support",
  labels=[("module", "goal_calculation"), ("data_type", "inputs")]
);

-- Create user_goals_enhanced table
CREATE TABLE IF NOT EXISTS `fiscal-fox-fin.fiscal_master_dw.user_goals_enhanced` (
  master_uid STRING NOT NULL,
  goal_id STRING NOT NULL,
  goal_type STRING NOT NULL,
  goal_name STRING,
  goal_description STRING,
  
  -- Timeline Goal Parameters
  target_date DATE,
  months_to_target INT64,
  required_monthly_payment FLOAT64,
  
  -- Investment Goal Parameters
  target_amount FLOAT64,
  horizon_years FLOAT64,
  expected_return_rate FLOAT64,
  required_monthly_sip FLOAT64,
  
  -- Emergency Fund Parameters
  months_of_coverage INT64,
  emergency_target_calculated FLOAT64,
  
  -- Feasibility Analysis
  feasible_with_current_surplus BOOLEAN,
  surplus_remaining_after_goal FLOAT64,
  feasibility_confidence_score FLOAT64,
  
  -- AI Enhancements
  ai_feasibility_score FLOAT64,
  ai_recommended_timeline_months INT64,
  ai_risk_assessment STRING,
  ai_optimization_suggestions JSON,
  
  -- Progress Tracking
  status STRING DEFAULT 'active',
  current_progress_amount FLOAT64,
  current_progress_percentage FLOAT64,
  last_progress_update DATE,
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(created_at)
CLUSTER BY master_uid, goal_type, status
OPTIONS(
  description="Enhanced goals with AI predictions for Fiscal Fox",
  labels=[("module", "goal_planning"), ("ai_enhanced", "true")]
);
EOF
