-- Net Worth Analysis Results Table
CREATE TABLE IF NOT EXISTS `fiscal-fox-in.fiscal_master_dw.networth_results` (
  master_uid STRING NOT NULL OPTIONS(description="User identifier from user_master table"),
  analysis_id STRING NOT NULL OPTIONS(description="Unique net worth analysis run identifier"),
  analysis_timestamp TIMESTAMP NOT NULL OPTIONS(description="When net worth analysis was performed"),
  
  -- Core net worth metrics
  net_worth FLOAT64 OPTIONS(description="Current net worth in INR"),
  total_assets FLOAT64 OPTIONS(description="Total assets in INR"),
  total_liabilities FLOAT64 OPTIONS(description="Total liabilities in INR"),
  credit_score INT64 OPTIONS(description="Credit score from credit report"),
  
  -- Net worth ratios
  debt_to_asset_ratio FLOAT64 OPTIONS(description="Debt to asset ratio (0-1)"),
  credit_utilization FLOAT64 OPTIONS(description="Credit utilization ratio (0-1)"),
  investment_ratio FLOAT64 OPTIONS(description="Investment to total assets ratio"),
  liquidity_ratio FLOAT64 OPTIONS(description="Liquid assets to liabilities ratio"),
  
  -- Analysis quality metrics
  data_quality_score FLOAT64 OPTIONS(description="Data quality score (0-100)"),
  validation_errors JSON OPTIONS(description="List of validation errors encountered"),
  missing_data_fields JSON OPTIONS(description="List of missing data fields"),
  
  -- Net worth analysis results
  investment_performance JSON OPTIONS(description="Investment performance breakdown"),
  rule_based_predictions JSON OPTIONS(description="Rule-based net worth predictions"),
  ml_predictions JSON OPTIONS(description="ML net worth predictions"),
  edge_case_test_results JSON OPTIONS(description="Edge case testing results"),
  raw_analysis_data JSON OPTIONS(description="Complete raw net worth analysis data"),
  
  -- Audit fields
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
PARTITION BY DATE(analysis_timestamp)
OPTIONS(
  description="Stores comprehensive net worth analysis results for all users",
  labels=[("team", "networth-analysis"), ("env", "prod")]
);
