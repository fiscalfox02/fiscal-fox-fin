#!/usr/bin/env python3
"""
Fiscal Fox Data Processing Script
Extracts JSON data from raw tables and creates structured processed tables
"""

import os
from google.cloud import bigquery
import json
from datetime import datetime

# Configuration
PROJECT_ID = "fiscal-fox-fin"
DATASET_ID = "fiscal_master_dw"
MASTER_UID = "ff_user_8a838f3528819407"

class FiscalFoxDataProcessor:
    def __init__(self):
        os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT_ID
        self.client = bigquery.Client(project=PROJECT_ID)
        self.master_uid = MASTER_UID
        
    def create_processed_tables(self):
        """Create all processed tables with proper schemas"""
        
        tables_sql = {
            'net_worth_processed': """
                CREATE TABLE IF NOT EXISTS `{project}.{dataset}.net_worth_processed` (
                    master_uid STRING NOT NULL,
                    asset_type STRING,
                    asset_value FLOAT64,
                    liability_type STRING,
                    liability_value FLOAT64,
                    total_net_worth FLOAT64,
                    account_id STRING,
                    account_type STRING,
                    bank_name STRING,
                    account_balance FLOAT64,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    source_created_at TIMESTAMP
                )
            """,
            
            'credit_report_processed': """
                CREATE TABLE IF NOT EXISTS `{project}.{dataset}.credit_report_processed` (
                    master_uid STRING NOT NULL,
                    credit_score INT64,
                    report_date STRING,
                    total_outstanding_balance FLOAT64,
                    total_past_due FLOAT64,
                    account_count INT64,
                    subscriber_name STRING,
                    account_type STRING,
                    current_balance FLOAT64,
                    amount_past_due FLOAT64,
                    payment_rating STRING,
                    interest_rate FLOAT64,
                    account_status STRING,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    source_created_at TIMESTAMP
                )
            """,
            
            'mf_transactions_processed': """
                CREATE TABLE IF NOT EXISTS `{project}.{dataset}.mf_transactions_processed` (
                    master_uid STRING NOT NULL,
                    isin_number STRING,
                    scheme_name STRING,
                    transaction_type STRING,
                    transaction_date TIMESTAMP,
                    transaction_amount FLOAT64,
                    transaction_units FLOAT64,
                    purchase_price FLOAT64,
                    folio_id STRING,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    source_created_at TIMESTAMP
                )
            """,
            
            'epf_holdings_processed': """
                CREATE TABLE IF NOT EXISTS `{project}.{dataset}.epf_holdings_processed` (
                    master_uid STRING NOT NULL,
                    establishment_name STRING,
                    member_id STRING,
                    office STRING,
                    date_of_joining DATE,
                    date_of_exit DATE,
                    pf_balance FLOAT64,
                    employee_share FLOAT64,
                    employer_share FLOAT64,
                    pension_balance FLOAT64,
                    total_pf_balance FLOAT64,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
                    source_created_at TIMESTAMP
                )
            """
        }
        
        print("🔧 Creating processed tables...")
        for table_name, sql in tables_sql.items():
            try:
                formatted_sql = sql.format(project=PROJECT_ID, dataset=DATASET_ID)
                query_job = self.client.query(formatted_sql)
                query_job.result()
                print(f"Created table: {table_name}")
            except Exception as e:
                print(f" Failed to create {table_name}: {e}")

    def safe_parse_currency(self, currency_obj):
        """Safely parse currency objects"""
        try:
            if not currency_obj:
                return 0.0
            units = float(currency_obj.get('units', 0))
            nanos = float(currency_obj.get('nanos', 0)) / 1e9
            return units + nanos
        except:
            return 0.0

    def process_net_worth_data(self):
        """Process net worth raw data"""
        print("\nProcessing Net Worth Data...")
        
        # Get raw data
        query = f"""
            SELECT json_data, created_at
            FROM `{PROJECT_ID}.{DATASET_ID}.net_worth_raw`
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        try:
            results = list(self.client.query(query).result())
            if not results:
                print("No net worth data found")
                return
            
            raw_data = results[0].json_data
            source_created_at = results[0].created_at
            
            # Parse JSON
            if isinstance(raw_data, str):
                data = json.loads(raw_data)
            else:
                data = dict(raw_data)
            
            processed_rows = []
            
            # Process assets
            net_worth_response = data.get('netWorthResponse', {})
            asset_values = net_worth_response.get('assetValues', [])
            
            for asset in asset_values:
                asset_type = asset.get('netWorthAttribute', '').replace('ASSET_TYPE_', '')
                asset_value = self.safe_parse_currency(asset.get('value', {}))
                
                processed_rows.append({
                    'master_uid': self.master_uid,
                    'asset_type': asset_type,
                    'asset_value': asset_value,
                    'liability_type': None,
                    'liability_value': None,
                    'total_net_worth': self.safe_parse_currency(net_worth_response.get('totalNetWorthValue', {})),
                    'account_id': None,
                    'account_type': None,
                    'bank_name': None,
                    'account_balance': None,
                    'source_created_at': source_created_at
                })
            
            # Process liabilities
            liability_values = net_worth_response.get('liabilityValues', [])
            for liability in liability_values:
                liability_type = liability.get('netWorthAttribute', '').replace('LIABILITY_TYPE_', '')
                liability_value = self.safe_parse_currency(liability.get('value', {}))
                
                processed_rows.append({
                    'master_uid': self.master_uid,
                    'asset_type': None,
                    'asset_value': None,
                    'liability_type': liability_type,
                    'liability_value': liability_value,
                    'total_net_worth': self.safe_parse_currency(net_worth_response.get('totalNetWorthValue', {})),
                    'account_id': None,
                    'account_type': None,
                    'bank_name': None,
                    'account_balance': None,
                    'source_created_at': source_created_at
                })
            
            # Process account details
            account_details = data.get('accountDetailsBulkResponse', {}).get('accountDetailsMap', {})
            for account_id, account_info in account_details.items():
                account_detail = account_info.get('accountDetails', {})
                deposit_summary = account_info.get('depositSummary', {})
                
                processed_rows.append({
                    'master_uid': self.master_uid,
                    'asset_type': None,
                    'asset_value': None,
                    'liability_type': None,
                    'liability_value': None,
                    'total_net_worth': self.safe_parse_currency(net_worth_response.get('totalNetWorthValue', {})),
                    'account_id': account_id,
                    'account_type': account_detail.get('accInstrumentType', ''),
                    'bank_name': account_detail.get('fipMeta', {}).get('name', ''),
                    'account_balance': self.safe_parse_currency(deposit_summary.get('currentBalance', {})),
                    'source_created_at': source_created_at
                })
            
            # Insert processed data
            table_id = f"{PROJECT_ID}.{DATASET_ID}.net_worth_processed"
            errors = self.client.insert_rows_json(table_id, processed_rows)
            
            if errors:
                print(f" Insert errors: {errors}")
            else:
                print(f"Processed {len(processed_rows)} net worth records")
                
        except Exception as e:
            print(f"Error processing net worth data: {e}")

    def process_credit_report_data(self):
        """Process credit report raw data"""
        print("\nProcessing Credit Report Data...")
        
        query = f"""
            SELECT json_data, created_at
            FROM `{PROJECT_ID}.{DATASET_ID}.credit_report_raw`
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        try:
            results = list(self.client.query(query).result())
            if not results:
                print("❌ No credit report data found")
                return
            
            raw_data = results[0].json_data
            source_created_at = results[0].created_at
            
            # Parse JSON
            if isinstance(raw_data, str):
                data = json.loads(raw_data)
            else:
                data = dict(raw_data)
            
            processed_rows = []
            
            # Extract credit report data
            credit_reports = data.get('creditReports', [])
            if credit_reports:
                credit_data = credit_reports[0].get('creditReportData', {})
                
                # Basic credit info
                credit_score = int(credit_data.get('score', {}).get('bureauScore', 0))
                report_date = credit_data.get('creditProfileHeader', {}).get('reportDate', '')
                
                # Credit account summary
                account_summary = credit_data.get('creditAccount', {}).get('creditAccountSummary', {})
                total_outstanding = float(account_summary.get('totalOutstandingBalance', {}).get('outstandingBalanceAll', 0))
                
                # Process individual accounts
                account_details = credit_data.get('creditAccount', {}).get('creditAccountDetails', [])
                
                total_past_due = 0
                for account in account_details:
                    current_balance = float(account.get('currentBalance', 0))
                    past_due = float(account.get('amountPastDue', 0))
                    total_past_due += past_due
                    
                    processed_rows.append({
                        'master_uid': self.master_uid,
                        'credit_score': credit_score,
                        'report_date': report_date,
                        'total_outstanding_balance': total_outstanding,
                        'total_past_due': total_past_due,
                        'account_count': len(account_details),
                        'subscriber_name': account.get('subscriberName', ''),
                        'account_type': account.get('accountType', ''),
                        'current_balance': current_balance,
                        'amount_past_due': past_due,
                        'payment_rating': account.get('paymentRating', ''),
                        'interest_rate': float(account.get('rateOfInterest', 0)),
                        'account_status': account.get('accountStatus', ''),
                        'source_created_at': source_created_at
                    })
                
                # Insert processed data
                table_id = f"{PROJECT_ID}.{DATASET_ID}.credit_report_processed"
                errors = self.client.insert_rows_json(table_id, processed_rows)
                
                if errors:
                    print(f"Insert errors: {errors}")
                else:
                    print(f" Processed {len(processed_rows)} credit report records")
                    
        except Exception as e:
            print(f" Error processing credit report data: {e}")

    def process_mf_transactions_data(self):
        """Process mutual fund transactions raw data"""
        print("\n Processing MF Transactions Data...")
        
        query = f"""
            SELECT json_data, created_at
            FROM `{PROJECT_ID}.{DATASET_ID}.mf_transactions_raw`
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        try:
            results = list(self.client.query(query).result())
            if not results:
                print("❌ No MF transactions data found")
                return
            
            raw_data = results[0].json_data
            source_created_at = results[0].created_at
            
            # Parse JSON
            if isinstance(raw_data, str):
                data = json.loads(raw_data)
            else:
                data = dict(raw_data)
            
            processed_rows = []
            
            # Process transactions
            transactions = data.get('transactions', [])
            for transaction in transactions:
                # Parse transaction date
                transaction_date = transaction.get('transactionDate', '')
                if transaction_date:
                    # Convert to timestamp
                    try:
                        transaction_timestamp = datetime.fromisoformat(transaction_date.replace('Z', '+00:00'))
                    except:
                        transaction_timestamp = None
                else:
                    transaction_timestamp = None
                
                processed_rows.append({
                    'master_uid': self.master_uid,
                    'isin_number': transaction.get('isinNumber', ''),
                    'scheme_name': transaction.get('schemeName', ''),
                    'transaction_type': transaction.get('externalOrderType', ''),
                    'transaction_date': transaction_timestamp,
                    'transaction_amount': self.safe_parse_currency(transaction.get('transactionAmount', {})),
                    'transaction_units': float(transaction.get('transactionUnits', 0)),
                    'purchase_price': self.safe_parse_currency(transaction.get('purchasePrice', {})),
                    'folio_id': transaction.get('folioId', ''),
                    'source_created_at': source_created_at
                })
            
            # Insert processed data
            table_id = f"{PROJECT_ID}.{DATASET_ID}.mf_transactions_processed"
            errors = self.client.insert_rows_json(table_id, processed_rows)
            
            if errors:
                print(f"Insert errors: {errors}")
            else:
                print(f"Processed {len(processed_rows)} MF transaction records")
                
        except Exception as e:
            print(f"Error processing MF transactions data: {e}")

    def process_epf_holdings_data(self):
        """Process EPF holdings raw data"""
        print("\n🏦 Processing EPF Holdings Data...")
        
        query = f"""
            SELECT json_data, created_at
            FROM `{PROJECT_ID}.{DATASET_ID}.epf_holdings_raw`
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        try:
            results = list(self.client.query(query).result())
            if not results:
                print("❌ No EPF holdings data found")
                return
            
            raw_data = results[0].json_data
            source_created_at = results[0].created_at
            
            # Parse JSON
            if isinstance(raw_data, str):
                data = json.loads(raw_data)
            else:
                data = dict(raw_data)
            
            processed_rows = []
            
            # Process UAN accounts
            uan_accounts = data.get('uanAccounts', [])
            for uan_account in uan_accounts:
                raw_details = uan_account.get('rawDetails', {})
                
                # Overall PF balance
                overall_balance = raw_details.get('overall_pf_balance', {})
                pension_balance = float(overall_balance.get('pension_balance', 0))
                current_pf_balance = float(overall_balance.get('current_pf_balance', 0))
                
                # Individual establishment details
                est_details = raw_details.get('est_details', [])
                for establishment in est_details:
                    # Parse dates
                    doj_epf = establishment.get('doj_epf', '')
                    doe_epf = establishment.get('doe_epf', '')
                    
                    # Convert date format from DD-MM-YYYY to YYYY-MM-DD
                    try:
                        if doj_epf:
                            doj_parts = doj_epf.split('-')
                            doj_formatted = f"{doj_parts[2]}-{doj_parts[1]}-{doj_parts[0]}"
                        else:
                            doj_formatted = None
                            
                        if doe_epf:
                            doe_parts = doe_epf.split('-')
                            doe_formatted = f"{doe_parts[2]}-{doe_parts[1]}-{doe_parts[0]}"
                        else:
                            doe_formatted = None
                    except:
                        doj_formatted = None
                        doe_formatted = None
                    
                    pf_balance_info = establishment.get('pf_balance', {})
                    
                    processed_rows.append({
                        'master_uid': self.master_uid,
                        'establishment_name': establishment.get('est_name', ''),
                        'member_id': establishment.get('member_id', ''),
                        'office': establishment.get('office', ''),
                        'date_of_joining': doj_formatted,
                        'date_of_exit': doe_formatted,
                        'pf_balance': float(pf_balance_info.get('net_balance', 0)),
                        'employee_share': float(pf_balance_info.get('employee_share', {}).get('balance', 0)),
                        'employer_share': float(pf_balance_info.get('employer_share', {}).get('balance', 0)),
                        'pension_balance': pension_balance,
                        'total_pf_balance': current_pf_balance,
                        'source_created_at': source_created_at
                    })
            
            # Insert processed data
            table_id = f"{PROJECT_ID}.{DATASET_ID}.epf_holdings_processed"
            errors = self.client.insert_rows_json(table_id, processed_rows)
            
            if errors:
                print(f"Insert errors: {errors}")
            else:
                print(f" Processed {len(processed_rows)} EPF holdings records")
                
        except Exception as e:
            print(f"Error processing EPF holdings data: {e}")

    def run_complete_processing(self):
        """Run complete data processing pipeline"""
        print("Starting Fiscal Fox Data Processing Pipeline")
        print("=" * 60)
        
        # Create tables
        self.create_processed_tables()
        
        # Process all data types
        self.process_net_worth_data()
        self.process_credit_report_data()
        self.process_mf_transactions_data()
        self.process_epf_holdings_data()
        
        print("\n✅ Data processing completed!")
        print(f"📊 All data processed with master_uid: {self.master_uid}")

if __name__ == "__main__":
    processor = FiscalFoxDataProcessor()
    processor.run_complete_processing()
