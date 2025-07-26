from google.cloud import bigquery
from google.oauth2 import service_account
import json
import os
from datetime import datetime
from typing import List, Dict, Optional

class BigQueryManager:
    def __init__(self, project_id: str = None, dataset_id: str = 'debt_analytics'):
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.dataset_id = dataset_id
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> bigquery.Client:
        """Initialize BigQuery client with proper authentication"""
        try:
            if os.path.exists('service-account-key.json'):
                credentials = service_account.Credentials.from_service_account_file(
                    'service-account-key.json'
                )
                return bigquery.Client(credentials=credentials, project=self.project_id)
            return bigquery.Client(project=self.project_id)
        except Exception as e:
            print(f" Error initializing BigQuery client: {e}")
            raise
    
    def save_debt_portfolio(self, user_id: str, debt_portfolio: List[Dict]) -> bool:
        """Save debt portfolio to BigQuery"""
        table_name = 'debt_portfolio'
        
        try:
            rows_to_insert = []
            for debt in debt_portfolio:
                row = {
                    'user_id': user_id,
                    'debt_id': f"{user_id}_{debt['debt_name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}",
                    'debt_name': debt['debt_name'],
                    'debt_type': debt['debt_type'],
                    'total_outstanding': float(debt['total_outstanding']),
                    'current_balance': float(debt['current_balance']),
                    'past_due': float(debt['past_due']),
                    'interest_rate': float(debt['interest_rate']),
                    'min_payment': float(debt['min_payment']),
                    'credit_limit': float(debt['credit_limit']),
                    'portfolio_type': debt['portfolio_type'],
                    'payment_rating': int(debt['payment_rating']),
                    'account_status': debt.get('account_status', 'Active'),
                    'payment_history': debt.get('payment_history', 'Regular'),
                    'created_timestamp': datetime.now().isoformat(),
                    'updated_timestamp': datetime.now().isoformat()
                }
                rows_to_insert.append(row)
            
            return self._insert_rows(table_name, rows_to_insert)
            
        except Exception as e:
            print(f" Error saving debt portfolio: {e}")
            return False
    
    def _insert_rows(self, table_name: str, rows: List[Dict], retries: int = 3) -> bool:
        """Insert rows with retry logic"""
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        for attempt in range(retries):
            try:
                table = self.client.get_table(table_id)
                errors = self.client.insert_rows_json(table, rows)
                
                if errors:
                    print(f"BigQuery insert errors (attempt {attempt + 1}): {errors}")
                    if attempt == retries - 1:
                        return False
                    continue
                else:
                    print(f"Successfully inserted {len(rows)} rows into {table_name}")
                    return True
                    
            except Exception as e:
                print(f" Error inserting rows (attempt {attempt + 1}): {e}")
                if attempt == retries - 1:
                    return False
        
        return False
