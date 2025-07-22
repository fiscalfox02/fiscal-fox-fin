# scripts/utilities/unified_data_loader.py
import json
import hashlib
import uuid
from datetime import datetime
from google.cloud import bigquery
from google.cloud import firestore

class FiscalFoxUnifiedLoader:
    def __init__(self, project_id):
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        self.firestore_client = firestore.Client(project=project_id)
        
    def generate_master_uid(self):
        """Generate consistent UID from your financial data fingerprint"""
        
        # Load all your JSON files
        with open('data/fetch_credit_report.json', 'r') as f:
            credit_data = json.load(f)
        with open('data/fetch_net_worth.json', 'r') as f:
            networth_data = json.load(f)
        with open('data/fetch_epf_details.json', 'r') as f:
            epf_data = json.load(f)
        with open('data/fetch_mf_transactions.json', 'r') as f:
            mf_data = json.load(f)
        
        # Create unique fingerprint from key financial data points
        bureau_score = credit_data['creditReports'][0]['creditReportData']['score']['bureauScore']
        total_networth = networth_data['netWorthResponse']['totalNetWorthValue']['units']
        epf_balance = epf_data['uanAccounts'][0]['rawDetails']['overall_pf_balance']['current_pf_balance']
        mf_count = len(mf_data['transactions'])
        
        # Create deterministic UID
        fingerprint = f"fiscal_fox_{bureau_score}_{total_networth}_{epf_balance}_{mf_count}"
        uid_hash = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        
        master_uid = f"ff_user_{uid_hash}"
        print(f"🔑 Generated Master UID: {master_uid}")
        
        return master_uid, {
            'credit_data': credit_data,
            'networth_data': networth_data,
            'epf_data': epf_data,
            'mf_data': mf_data
        }
    
    def load_all_data(self):
        """Load all financial data with unified UID across all modules"""
        
        # Generate consistent UID
        master_uid, financial_data = self.generate_master_uid()
        
        print(f"Loading data for UID: {master_uid}")
        
        # Load data into BigQuery (analytics)
        self.load_bigquery_data(master_uid, financial_data)
        
        # Load data into Firestore (real-time)
        self.load_firestore_data(master_uid, financial_data)
        
        print(" All data loaded successfully across all modules!")
        return master_uid
    
    def load_bigquery_data(self, master_uid, financial_data):
        """Load data into BigQuery for all modules"""
        
        print("Loading BigQuery data...")
        
        # 1. Debt Optimizer Data
        self.load_debt_data_bq(master_uid, financial_data['credit_data'])
        
        # 2. Net Worth Forecaster Data  
        self.load_networth_data_bq(master_uid, financial_data['networth_data'])
        
        # 3. EPF Holdings Data
        self.load_epf_data_bq(master_uid, financial_data['epf_data'])
        
        # 4. Mutual Fund Data
        self.load_mf_data_bq(master_uid, financial_data['mf_data'])
        
        # 5. Master User Registry
        self.register_user_bq(master_uid)
    
    def load_debt_data_bq(self, master_uid, credit_data):
        """Load credit report into debt_accounts table"""
        
        accounts = credit_data['creditReports'][0]['creditReportData']['creditAccount']['creditAccountDetails']
        
        rows = []
        for i, account in enumerate(accounts):
            if float(account.get('currentBalance', 0)) > 0:
                
                balance = float(account['currentBalance'])
                interest_rate = float(account['rateOfInterest'])
                monthly_interest = balance * interest_rate / 1200
                
                # Calculate optimal payment strategy
                if interest_rate >= 25:
                    optimal_payment = balance * 0.15
                    strategy = 'avalanche'
                elif interest_rate >= 18:
                    optimal_payment = balance * 0.10
                    strategy = 'hybrid'
                else:
                    optimal_payment = balance * 0.05
                    strategy = 'snowball'
                
                rows.append({
                    'master_uid': master_uid,
                    'account_id': f"{account['subscriberName']}_{i}",
                    'subscriber_name': account['subscriberName'],
                    'account_type': account['accountType'],
                    'current_balance': balance,
                    'rate_of_interest': interest_rate,
                    'amount_past_due': float(account.get('amountPastDue', 0)),
                    'payment_rating': int(account.get('paymentRating', 0)),
                    'account_status': account['accountStatus'],
                    'credit_limit': float(account.get('highestCreditOrOriginalLoanAmount', 0)),
                    'monthly_interest_cost': monthly_interest,
                    'optimal_payment': optimal_payment,
                    'strategy_type': strategy,
                    'priority_rank': i + 1,
                    'created_at': datetime.now()
                })
        
        if rows:
            table_id = f"{self.project_id}.fiscal_master_dw.debt_accounts"
            errors = self.bq_client.insert_rows_json(table_id, rows)
            if not errors:
                print(f"✅ Loaded {len(rows)} debt accounts")
    
    def load_networth_data_bq(self, master_uid, networth_data):
        """Load net worth data into networth_snapshots table"""
        
        response = networth_data['netWorthResponse']
        
        # Extract asset values
        assets = {}
        for asset in response['assetValues']:
            assets[asset['netWorthAttribute']] = float(asset['value']['units'])
        
        # Extract liability values  
        liabilities = {}
        total_liabilities = 0
        for liability in response['liabilityValues']:
            value = float(liability['value']['units'])
            liabilities[liability['netWorthAttribute']] = value
            total_liabilities += value
        
        total_networth = float(response['totalNetWorthValue']['units'])
        
        # Simple forecasting (8% annual growth)
        monthly_growth = 0.08 / 12
        forecast_3m = total_networth * ((1 + monthly_growth) ** 3)
        forecast_6m = total_networth * ((1 + monthly_growth) ** 6)
        forecast_12m = total_networth * ((1 + monthly_growth) ** 12)
        
        row = [{
            'master_uid': master_uid,
            'snapshot_date': datetime.now(),
            'total_net_worth': total_networth,
            'total_assets': sum(assets.values()),
            'total_liabilities': total_liabilities,
            'asset_breakdown': assets,
            'liability_breakdown': liabilities,
            'epf_value': assets.get('ASSET_TYPE_EPF', 0),
            'mf_value': assets.get('ASSET_TYPE_MUTUAL_FUND', 0),
            'securities_value': assets.get('ASSET_TYPE_INDIAN_SECURITIES', 0),
            'savings_value': assets.get('ASSET_TYPE_SAVINGS_ACCOUNTS', 0),
            'forecast_3m': forecast_3m,
            'forecast_6m': forecast_6m,
            'forecast_12m': forecast_12m,
            'created_at': datetime.now()
        }]
        
        table_id = f"{self.project_id}.fiscal_master_dw.networth_snapshots"
        errors = self.bq_client.insert_rows_json(table_id, row)
        if not errors:
            print("✅ Loaded net worth snapshot")
    
    def load_firestore_data(self, master_uid, financial_data):
        """Load data into Firestore for real-time access"""
        
        print("🔥 Loading Firestore data...")
        
        # Create master user document
        user_doc_ref = self.firestore_client.collection('users').document(master_uid)
        
        # Calculate summary data
        credit_data = financial_data['credit_data']
        networth_data = financial_data['networth_data']
        
        # Debt summary
        debt_accounts = credit_data['creditReports'][0]['creditReportData']['creditAccount']['creditAccountDetails']
        total_debt = sum(float(acc.get('currentBalance', 0)) for acc in debt_accounts)
        monthly_interest = sum(
            float(acc.get('currentBalance', 0)) * float(acc.get('rateOfInterest', 0)) / 1200 
            for acc in debt_accounts
        )
        
        # Net worth summary
        total_networth = float(networth_data['netWorthResponse']['totalNetWorthValue']['units'])
        
        # Store master profile
        user_profile = {
            'master_uid': master_uid,
            'created_at': firestore.SERVER_TIMESTAMP,
            'last_updated': firestore.SERVER_TIMESTAMP,
            'data_status': 'loaded',
            
            # Module enablement
            'modules': {
                'debt_optimizer': {'enabled': True, 'last_sync': firestore.SERVER_TIMESTAMP},
                'networth_forecaster': {'enabled': True, 'last_sync': firestore.SERVER_TIMESTAMP},
                'goal_engine': {'enabled': True, 'last_sync': firestore.SERVER_TIMESTAMP},
                'ask_fin': {'enabled': True, 'last_sync': firestore.SERVER_TIMESTAMP}
            },
            
            # Financial summary
            'financial_summary': {
                'total_debt': total_debt,
                'monthly_interest_cost': monthly_interest,
                'total_net_worth': total_networth,
                'debt_accounts_count': len([acc for acc in debt_accounts if float(acc.get('currentBalance', 0)) > 0])
            }
        }
        
        user_doc_ref.set(user_profile)
        print(f"✅ Created Firestore user profile for {master_uid}")

if __name__ == "__main__":
    import os
    
    project_id = "fiscal-fox-fin"  # Your project ID
    loader = FiscalFoxUnifiedLoader(project_id)
    master_uid = loader.load_all_data()
    print(f"\n Your Master UID: {master_uid}")
    print("📋 Save this UID - you'll use it to test all modules!")
