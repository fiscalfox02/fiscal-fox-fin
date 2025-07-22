PROJECT_ID="fiscal-fox-fin"
DATASET_ID="fiscal_master_dw"

echo "🏗️ Creating BigQuery tables for all Fiscal Fox modules..."

# Debt Optimizer table
bq mk --table \
  --description "Debt accounts for optimization analysis" \
  "$PROJECT_ID:$DATASET_ID.debt_accounts" \
  master_uid:STRING,account_id:STRING,subscriber_name:STRING,account_type:STRING,current_balance:FLOAT64,rate_of_interest:FLOAT64,amount_past_due:FLOAT64,payment_rating:INT64,account_status:STRING,credit_limit:FLOAT64,monthly_interest_cost:FLOAT64,optimal_payment:FLOAT64,strategy_type:STRING,priority_rank:INT64,created_at:TIMESTAMP

# Net Worth Forecaster table
bq mk --table \
  --description "Net worth snapshots and forecasts" \
  "$PROJECT_ID:$DATASET_ID.networth_snapshots" \
  master_uid:STRING,snapshot_date:TIMESTAMP,total_net_worth:FLOAT64,total_assets:FLOAT64,total_liabilities:FLOAT64,asset_breakdown:JSON,liability_breakdown:JSON,epf_value:FLOAT64,mf_value:FLOAT64,securities_value:FLOAT64,savings_value:FLOAT64,forecast_3m:FLOAT64,forecast_6m:FLOAT64,forecast_12m:FLOAT64,created_at:TIMESTAMP

# Goal Engine table
bq mk --table \
  --description "User financial goals and tracking" \
  "$PROJECT_ID:$DATASET_ID.user_goals" \
  master_uid:STRING,goal_id:STRING,goal_type:STRING,goal_name:STRING,target_amount:FLOAT64,current_progress:FLOAT64,target_date:DATE,monthly_allocation:FLOAT64,feasibility_score:FLOAT64,priority:STRING,status:STRING,created_at:TIMESTAMP

# Ask Fin conversation table
bq mk --table \
  --description "AI advisor conversations and context" \
  "$PROJECT_ID:$DATASET_ID.ask_fin_conversations" \
  master_uid:STRING,conversation_id:STRING,user_query:STRING,ai_response:STRING,context_modules:STRING,user_context:JSON,timestamp:TIMESTAMP

# EPF holdings table
bq mk --table \
  --description "EPF account details and balances" \
  "$PROJECT_ID:$DATASET_ID.epf_holdings" \
  master_uid:STRING,establishment_name:STRING,member_id:STRING,total_balance:FLOAT64,employee_share:FLOAT64,employer_share:FLOAT64,doj_epf:DATE,doe_epf:DATE,created_at:TIMESTAMP

# Mutual fund transactions table
bq mk --table \
  --description "Mutual fund transaction history" \
  "$PROJECT_ID:$DATASET_ID.mf_transactions" \
  master_uid:STRING,transaction_id:STRING,isin_number:STRING,folio_id:STRING,scheme_name:STRING,transaction_type:STRING,transaction_date:TIMESTAMP,transaction_amount:FLOAT64,transaction_units:FLOAT64,purchase_price:FLOAT64,created_at:TIMESTAMP

# Master user registry
bq mk --table \
  --description "Master user registry for all modules" \
  "$PROJECT_ID:$DATASET_ID.users_master" \
  master_uid:STRING,active_modules:STRING,data_fingerprint:STRING,created_at:TIMESTAMP,last_active:TIMESTAMP,status:STRING

echo "✅ All BigQuery tables created successfully!"
