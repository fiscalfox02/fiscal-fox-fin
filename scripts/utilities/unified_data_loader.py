# scripts/utilities/unified_data_loader.py - Fixed version
import json
import hashlib
from datetime import datetime

class FiscalFoxUnifiedLoader:
    def __init__(self):
        self.data_sources = {
            'credit_report': 'data/fetch_credit_report.json',
            'net_worth': 'data/fetch_net_worth.json', 
            'epf_details': 'data/fetch_epf_details.json',
            'mf_transactions': 'data/fetch_mf_transactions.json'
        }
    
    def generate_unified_uid(self):
        """Generate consistent UID from your actual financial data"""
        
        # Load all JSON files
        financial_data = {}
        for source_name, file_path in self.data_sources.items():
            with open(file_path, 'r') as f:
                financial_data[source_name] = json.load(f)
        
        # Extract key identifying elements from actual data structure
        identity_components = self._extract_identity_components(financial_data)
        
        # Generate deterministic UID
        master_uid = self._create_deterministic_uid(identity_components)
        
        return master_uid, identity_components, financial_data
    
    def _extract_identity_components(self, financial_data):
        """Extract key identifying components from actual JSON structure"""
        
        components = {}
        
        # 1. Credit Report Key Data
        credit_data = financial_data['credit_report']
        components['bureau_score'] = credit_data['creditReports'][0]['creditReportData']['score']['bureauScore']
        components['total_outstanding'] = credit_data['creditReports'][0]['creditReportData']['creditAccount']['creditAccountSummary']['totalOutstandingBalance']['outstandingBalanceAll']
        components['active_accounts'] = credit_data['creditReports'][0]['creditReportData']['creditAccount']['creditAccountSummary']['account']['creditAccountActive']
        
        # 2. Net Worth Key Data  
        networth_data = financial_data['net_worth']
        components['total_net_worth'] = networth_data['netWorthResponse']['totalNetWorthValue']['units']
        
        # 3. EPF Key Data
        epf_data = financial_data['epf_details']
        components['current_pf_balance'] = epf_data['uanAccounts'][0]['rawDetails']['overall_pf_balance']['current_pf_balance']
        components['epf_establishments'] = len(epf_data['uanAccounts'][0]['rawDetails']['est_details'])
        
        # 4. Mutual Fund Key Data
        mf_data = financial_data['mf_transactions']
        components['mf_transactions_count'] = len(mf_data['transactions'])
        
        # Calculate total MF investment amount
        total_mf_investment = 0
        for txn in mf_data['transactions']:
            units = float(txn['transactionAmount']['units'])
            nanos = float(txn['transactionAmount'].get('nanos', 0)) / 1e9
            total_mf_investment += units + nanos
        components['total_mf_investment'] = int(total_mf_investment)
        
        return components
    
    def _create_deterministic_uid(self, components):
        """Create deterministic UID from identity components"""
        
        fingerprint_elements = [
            f"bureau_{components['bureau_score']}",
            f"networth_{components['total_net_worth']}",
            f"pf_{components['current_pf_balance']}",
            f"debt_{components['total_outstanding']}",
            f"mf_{components['mf_transactions_count']}",
            f"epf_est_{components['epf_establishments']}"
        ]
        
        fingerprint_string = "_".join(fingerprint_elements)
        uid_hash = hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]
        master_uid = f"ff_user_{uid_hash}"
        
        return master_uid
    
    def get_extracted_data_for_bigquery(self, master_uid, financial_data):
        """Extract and structure data ready for BigQuery insertion"""
        
        extracted_data = {
            'master_uid': master_uid,
            'extraction_timestamp': datetime.now(),
            'debt_accounts': self._extract_debt_accounts(master_uid, financial_data['credit_report']),
            'networth_snapshot': self._extract_networth_snapshot(master_uid, financial_data['net_worth']),
            'epf_holdings': self._extract_epf_holdings(master_uid, financial_data['epf_details']),
            'mf_transactions': self._extract_mf_transactions(master_uid, financial_data['mf_transactions'])
        }
        
        return extracted_data
    
    def _extract_debt_accounts(self, master_uid, credit_data):
        """Extract debt accounts with robust error handling"""
        
        accounts = credit_data['creditReports'][0]['creditReportData']['creditAccount']['creditAccountDetails']
        debt_accounts = []
        
        for i, account in enumerate(accounts):
            try:
                current_balance = float(account.get('currentBalance', 0))
                
                if current_balance > 0:
                    interest_rate = float(account.get('rateOfInterest', 0))
                    
                    debt_account = {
                        'master_uid': master_uid,
                        'account_id': f"{account.get('subscriberName', f'Account_{i}')}_{i}",
                        'subscriber_name': account.get('subscriberName', f'Unknown_Lender_{i}'),
                        'account_type': account.get('accountType', 'unknown'),
                        'current_balance': current_balance,
                        'rate_of_interest': interest_rate,
                        'amount_past_due': float(account.get('amountPastDue', 0)),
                        'payment_rating': int(account.get('paymentRating', 0)),
                        'account_status': account.get('accountStatus', 'unknown'),
                        'credit_limit': float(account.get('highestCreditOrOriginalLoanAmount', 0)),
                        'monthly_interest_cost': current_balance * interest_rate / 1200 if interest_rate > 0 else 0,
                        'priority_rank': i + 1,
                        'open_date': account.get('openDate'),
                        'payment_history': account.get('paymentHistoryProfile', '')
                    }
                    
                    debt_accounts.append(debt_account)
                    print(f"✓ Processed debt: {account.get('subscriberName')} (₹{current_balance:,.0f})")
            
            except Exception as e:
                print(f" Error processing debt account {i}: {e}")
                continue
        
        return debt_accounts
    
    def _extract_networth_snapshot(self, master_uid, networth_data):
        """Extract net worth data with error handling"""
        
        try:
            # Assets breakdown
            assets = {}
            total_assets = 0
            for asset in networth_data['netWorthResponse']['assetValues']:
                asset_type = asset['netWorthAttribute']
                value = float(asset['value']['units'])
                assets[asset_type] = value
                total_assets += value
            
            # Liabilities breakdown
            liabilities = {}
            total_liabilities = 0
            for liability in networth_data['netWorthResponse']['liabilityValues']:
                liability_type = liability['netWorthAttribute']
                value = float(liability['value']['units'])
                liabilities[liability_type] = value
                total_liabilities += value
            
            total_networth = float(networth_data['netWorthResponse']['totalNetWorthValue']['units'])
            
            networth_snapshot = {
                'master_uid': master_uid,
                'snapshot_date': datetime.now(),
                'total_net_worth': total_networth,
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'mutual_fund_value': assets.get('ASSET_TYPE_MUTUAL_FUND', 0),
                'epf_value': assets.get('ASSET_TYPE_EPF', 0),
                'securities_value': assets.get('ASSET_TYPE_INDIAN_SECURITIES', 0),
                'savings_value': assets.get('ASSET_TYPE_SAVINGS_ACCOUNTS', 0),
                'asset_breakdown': assets,
                'liability_breakdown': liabilities
            }
            
            print(f"✓ Processed net worth: ₹{total_networth:,}")
            return networth_snapshot
            
        except Exception as e:
            print(f"Error processing net worth data: {e}")
            return {}
    
    def _extract_epf_holdings(self, master_uid, epf_data):
        """Extract EPF holdings with robust error handling for inconsistent data structure"""
        
        epf_holdings = []
        establishments = epf_data['uanAccounts'][0]['rawDetails']['est_details']
        
        for est in establishments:
            try:
                # Handle inconsistent pf_balance structure
                pf_balance = est.get('pf_balance', {})
                employee_share = pf_balance.get('employee_share', {})
                employer_share = pf_balance.get('employer_share', {})
                
                # Use credit as fallback for balance if balance is missing
                employee_balance = float(employee_share.get('balance', employee_share.get('credit', 0)))
                employer_balance = float(employer_share.get('balance', employer_share.get('credit', 0)))
                
                holding = {
                    'master_uid': master_uid,
                    'establishment_name': est.get('est_name', 'Unknown'),
                    'member_id': est.get('member_id', 'Unknown'),
                    'office': est.get('office', 'Unknown'),
                    'total_balance': float(pf_balance.get('net_balance', 0)),
                    'employee_share': employee_balance,
                    'employer_share': employer_balance,
                    'date_of_joining': est.get('doj_epf', ''),
                    'date_of_exit': est.get('doe_epf', ''),
                    'date_of_exit_eps': est.get('doe_eps', '')
                }
                
                epf_holdings.append(holding)
                print(f"✓ Processed EPF: {est.get('est_name')} (Balance: ₹{holding['total_balance']:,.0f})")
                
            except Exception as e:
                print(f"Error processing EPF establishment {est.get('est_name', 'Unknown')}: {e}")
                continue
        
        return epf_holdings
    
    def _extract_mf_transactions(self, master_uid, mf_data):
        """Extract MF transactions with error handling"""
        
        mf_transactions = []
        
        for i, txn in enumerate(mf_data['transactions']):
            try:
                # Handle units and nanos properly
                amount_units = float(txn['transactionAmount']['units'])
                amount_nanos = float(txn['transactionAmount'].get('nanos', 0)) / 1e9
                total_amount = amount_units + amount_nanos
                
                price_units = float(txn['purchasePrice']['units'])
                price_nanos = float(txn['purchasePrice'].get('nanos', 0)) / 1e9
                total_price = price_units + price_nanos
                
                transaction = {
                    'master_uid': master_uid,
                    'transaction_id': f"{txn['folioId']}_{i}",
                    'isin_number': txn['isinNumber'],
                    'folio_id': txn['folioId'],
                    'scheme_name': txn['schemeName'],
                    'transaction_type': txn['externalOrderType'],
                    'transaction_date': txn['transactionDate'],
                    'transaction_amount': total_amount,
                    'transaction_units': float(txn['transactionUnits']),
                    'purchase_price': total_price,
                    'transaction_mode': txn.get('transactionMode', 'N')
                }
                mf_transactions.append(transaction)
                print(f"✓ Processed MF: {txn['schemeName']} (₹{total_amount:,.0f})")
                
            except Exception as e:
                print(f" Error processing MF transaction {i}: {e}")
                continue
        
        return mf_transactions

def main():
    """Generate unified UID and extract data for BigQuery"""
    
    generator = FiscalFoxUnifiedLoader()
    
    print("🔄 Generating Unified UID from your financial data...")
    
    try:
        # Generate UID and extract components
        master_uid, identity_components, financial_data = generator.generate_unified_uid()
        
        print(f" Generated Master UID: {master_uid}")
        print(f"Identity Components:")
        for key, value in identity_components.items():
            print(f"   {key}: {value}")
        
        # Extract BigQuery-ready data
        extracted_data = generator.get_extracted_data_for_bigquery(master_uid, financial_data)
        
        print(f"\n📈 Extracted Data Summary:")
        print(f"   Debt Accounts: {len(extracted_data['debt_accounts'])}")
        print(f"   Net Worth: ₹{extracted_data['networth_snapshot'].get('total_net_worth', 0):,}")
        print(f"   EPF Holdings: {len(extracted_data['epf_holdings'])} establishments")
        print(f"   MF Transactions: {len(extracted_data['mf_transactions'])}")
        
        print(f"\nYour consistent Master UID: {master_uid}")
        print(" Use this UID across all Fiscal Fox modules!")
        
        return master_uid, extracted_data
        
    except Exception as e:
        print(f" Error during processing: {e}")
        return None, None

if __name__ == "__main__":
    main()

