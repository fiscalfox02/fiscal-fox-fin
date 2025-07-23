
-- User Financial Profile Table
-- Stores aggregated financial metrics for goal planning
-- Master UID: ff_user_8a838f3528819407 (placeholder for any user)

CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.fiscal_master_dw.user_financial_profile` (
  master_uid STRING NOT NULL,
  bureau_score INT64,                    -- Your: 746
  total_outstanding FLOAT64,             -- Your: 75000
  active_accounts INT64,                 -- Your: 6
  total_net_worth FLOAT64,               -- Your: 868721
  current_pf_balance FLOAT64,            -- Your: 211111
  epf_establishments INT64,              -- Your: 2
  mf_transactions_count INT64,           -- Your: 4
  total_mf_investment FLOAT64,           -- Your: 48553
  
  -- Calculated fields for goal setting
  monthly_income_estimate FLOAT64,       -- Estimated from net worth
  monthly_expenses_estimate FLOAT64,     -- Estimated lifestyle cost
  calculated_monthly_surplus FLOAT64,    -- Available for goal allocation
  
  -- Risk profiling for investment goals
  risk_profile STRING,                   -- 'conservative', 'moderate', 'aggressive'
  income_type STRING,                    -- 'salaried', 'gig', 'contract'
  expense_volatility FLOAT64,           -- 0.0-1.0 scale
  
  -- Metadata
  data_last_refreshed TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(created_at)
CLUSTER BY master_uid
OPTIONS(
  description="Financial profile for goal planning with Master UID as placeholder",
  labels=[("module", "goal_setting"), ("data_type", "user_profile")]
);
<<<<<<< HEAD
=======
EOF
>>>>>>> 5f9b84ac9dd82682d2b10455f8fdc38ffad3ba45
