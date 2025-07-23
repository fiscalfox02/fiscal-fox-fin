-- Net Worth Webhook Cache Table for Quick API Access
CREATE TABLE IF NOT EXISTS `fiscal-fox-in.fiscal_master_dw.networth_webhook` (
  master_uid STRING NOT NULL OPTIONS(description="User identifier"),
  latest_analysis_id STRING OPTIONS(description="Most recent net worth analysis ID"),
  latest_net_worth FLOAT64 OPTIONS(description="Latest net worth value"),
  latest_credit_score INT64 OPTIONS(description="Latest credit score"),
  latest_data_quality FLOAT64 OPTIONS(description="Latest data quality score"),
  analysis_count INT64 DEFAULT 0 OPTIONS(description="Total number of net worth analyses performed"),
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP() OPTIONS(description="Last cache update time"),
  
  -- Quick access fields for net worth API
  summary_json JSON OPTIONS(description="Net worth summary data for quick API responses"),
  
  -- Net worth trends (last 3 analyses)
  networth_trend JSON OPTIONS(description="Net worth trend data for quick charts"),
  
  -- Audit
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
)
OPTIONS(
  description="Cache table for webhook API quick access to latest net worth analysis results",
  labels=[("team", "networth-analysis"), ("type", "webhook-cache")]
);
