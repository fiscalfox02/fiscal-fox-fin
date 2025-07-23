-- Goal Calculation Inputs Table for Fiscal Fox
-- Preprocessed data for your goal setting algorithms
-- Master UID: ff_user_8a838f3528819407

CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.fiscal_master_dw.goal_calculation_inputs` (
  master_uid STRING NOT NULL,
  calculation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  
  -- Debt inputs for timeline goals (your ₹75,000 debt)
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
  
  -- Goal setting parameters (your ₹25,000 surplus)
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
