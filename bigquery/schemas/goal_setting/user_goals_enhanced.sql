-- Enhanced User Goals Table for Fiscal Fox
-- Stores goals with AI predictions, gamification, and your goal logic

CREATE TABLE IF NOT EXISTS `${PROJECT_ID}.fiscal_master_dw.user_goals_enhanced` (
  master_uid STRING NOT NULL,
  goal_id STRING NOT NULL,
  goal_type STRING NOT NULL,             -- 'timeline', 'recurring', 'emergency', 'investment', 'micro'
  goal_name STRING,
  goal_description STRING,
  
  -- Timeline Goal Parameters (your amortization logic)
  target_date DATE,
  months_to_target INT64,
  required_monthly_payment FLOAT64,      -- From your amortization calculation
  
  -- Investment Goal Parameters (your SIP logic)
  target_amount FLOAT64,
  horizon_years FLOAT64,
  expected_return_rate FLOAT64,          -- Your default 12%
  required_monthly_sip FLOAT64,          -- Your SIP calculation
  
  -- Emergency Fund Parameters (your emergency logic)
  months_of_coverage INT64,              -- 6-9 months based on income type
  emergency_target_calculated FLOAT64,   -- Based on expense + EMIs
  
  -- Recurring Investment Parameters (your risk profile logic)
  equity_allocation_percent FLOAT64,     -- Based on risk profile
  debt_allocation_percent FLOAT64,
  hybrid_allocation_percent FLOAT64,
  
  -- Micro Goal Parameters (your micro logic)
  weekly_target_amount FLOAT64,
  paydays_alignment INT64,
  
  -- Feasibility Analysis (your calculations)
  feasible_with_current_surplus BOOLEAN,
  surplus_remaining_after_goal FLOAT64,
  feasibility_confidence_score FLOAT64,
  
  -- AI Enhancements (for future Vertex AI integration)
  ai_feasibility_score FLOAT64,
  ai_recommended_timeline_months INT64,
  ai_risk_assessment STRING,
  ai_optimization_suggestions JSON,
  
  -- ML Model Predictions
  ml_success_probability FLOAT64,
  ml_optimal_allocation JSON,
  ml_scenario_outcomes JSON,
  
  -- Gamification Elements
  achievement_points INT64,
  milestone_markers JSON,
  completion_rewards JSON,
  
  -- Progress Tracking
  status STRING DEFAULT 'active',
  current_progress_amount FLOAT64,
  current_progress_percentage FLOAT64,
  last_progress_update DATE,
  
  -- Goal Dependencies
  conflicts_with_goals ARRAY<STRING>,
  depends_on_goals ARRAY<STRING>,
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(created_at)
CLUSTER BY master_uid, goal_type, status
OPTIONS(
  description="Enhanced goals with AI predictions and gamification for Fiscal Fox",
  labels=[("module", "goal_planning"), ("ai_enhanced", "true"), ("gamified", "true")]
);
