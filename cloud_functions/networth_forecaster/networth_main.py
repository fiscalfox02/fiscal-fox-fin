#!/usr/bin/env python3
"""
Fiscal Fox Net Worth Analyzer - Structured Tables Version
Works with normalized BigQuery tables (not JSON format)
"""

import json
import sys
import os
import hashlib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from google.cloud import bigquery
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from cryptography.fernet import Fernet
import warnings
from typing import Dict, List, Optional, Union
warnings.filterwarnings('ignore')


class FiscalFoxNetWorthAnalyzer:
    """Net Worth Analyzer for Structured BigQuery Tables"""
    
    def __init__(self, 
                 master_uid: str = "ff_user_8a838f3528819407", 
                 enable_privacy: bool = True,
                 use_local_files: bool = False,
                 local_data_path: str = "data/"):
        
        self.master_uid = master_uid
        self.project_id = "fiscal-fox-fin"
        self.dataset_id = "fiscal_master_dw"
        self.enable_privacy = enable_privacy
        self.use_local_files = use_local_files
        self.local_data_path = local_data_path
        
        # Core data structures
        self.data = {}
        self.financial_ratios = {}
        self.validation_errors = []
        self.missing_data_fields = []
        self.analysis_id = self._generate_analysis_id()
        
        # Security setup
        if enable_privacy:
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
        
        # Setup clients and logging
        if not use_local_files:
            self.client = self._setup_bigquery_client()
        else:
            self.client = None
            
        self._setup_logging()
        
        # Create data directory if using local files
        if use_local_files:
            os.makedirs(local_data_path, exist_ok=True)

    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.master_uid}_networth_analysis_{timestamp}"
    
    def _setup_bigquery_client(self):
        """Initialize BigQuery client"""
        try:
            client = bigquery.Client(project=self.project_id)
            print(f"✅ Connected to BigQuery project: {self.project_id}")
            return client
        except Exception as e:
            print(f"❌ Failed to connect to BigQuery: {e}")
            return None
    
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f'FiscalFoxAnalyzer_{self.master_uid}')

    def add_differential_privacy_noise(self, value: float, epsilon: float = 1.0) -> float:
        """Add Laplace noise for differential privacy"""
        if not self.enable_privacy:
            return value
        
        sensitivity = abs(value) * 0.01  # 1% sensitivity
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return max(0, value + noise)  # Ensure non-negative

    def load_user_data_from_bigquery(self) -> Dict:
        """Load user data from structured BigQuery tables"""
        if not self.client:
            raise Exception("BigQuery client not initialized")
        
        user_data = {}
        
        # 1. Load Net Worth Data (Assets & Liabilities)
        try:
            print(f"📊 Loading net worth data for {self.master_uid}...")
            
            networth_query = f"""
                SELECT 
                    asset_type,
                    asset_value,
                    liability_type, 
                    liability_value,
                    total_net_worth,
                    account_id,
                    account_type,
                    bank_name,
                    account_balance
                FROM `{self.project_id}.{self.dataset_id}.net_worth_processed`
                WHERE master_uid = @master_uid
                ORDER BY processed_at DESC
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("master_uid", "STRING", self.master_uid)
                ]
            )
            
            query_job = self.client.query(networth_query, job_config=job_config)
            results = query_job.result()
            
            # Process net worth data
            assets = {}
            liabilities = {}
            accounts = {}
            total_net_worth = 0
            
            for row in results:
                # Assets
                if row.asset_type and row.asset_value:
                    asset_key = row.asset_type.upper().replace(' ', '_')
                    assets[asset_key] = assets.get(asset_key, 0) + float(row.asset_value)
                
                # Liabilities  
                if row.liability_type and row.liability_value:
                    liability_key = row.liability_type.upper().replace(' ', '_')
                    liabilities[liability_key] = liabilities.get(liability_key, 0) + float(row.liability_value)
                
                # Account details
                if row.account_id:
                    accounts[row.account_id] = {
                        'account_type': row.account_type,
                        'bank_name': row.bank_name,
                        'balance': float(row.account_balance) if row.account_balance else 0
                    }
                
                # Net worth
                if row.total_net_worth:
                    total_net_worth = float(row.total_net_worth)
            
            user_data['net_worth'] = {
                'assets': assets,
                'liabilities': liabilities,
                'accounts': accounts,
                'total_net_worth': total_net_worth
            }
            
            print(f"✅ Loaded net worth data: {len(assets)} assets, {len(liabilities)} liabilities")
            
        except Exception as e:
            print(f"❌ Error loading net worth: {e}")
            user_data['net_worth'] = {}
            self.validation_errors.append(f"Failed to load net worth: {str(e)}")
        
        # 2. Load Credit Report Data
        try:
            print(f"📊 Loading credit data for {self.master_uid}...")
            
            credit_query = f"""
                SELECT 
                    credit_score,
                    total_outstanding_balance,
                    total_past_due,
                    account_count,
                    current_balance,
                    amount_past_due,
                    account_type,
                    payment_rating,
                    account_status
                FROM `{self.project_id}.{self.dataset_id}.credit_report_processed`
                WHERE master_uid = @master_uid
                ORDER BY processed_at DESC
            """
            
            query_job = self.client.query(credit_query, job_config=job_config)
            results = query_job.result()
            
            credit_accounts = []
            credit_score = 650
            total_outstanding = 0
            total_past_due = 0
            
            for row in results:
                if row.credit_score:
                    credit_score = int(row.credit_score)
                
                if row.total_outstanding_balance:
                    total_outstanding = float(row.total_outstanding_balance)
                
                if row.total_past_due:
                    total_past_due = float(row.total_past_due)
                
                # Individual account details
                if row.current_balance is not None:
                    credit_accounts.append({
                        'currentBalance': float(row.current_balance),
                        'amountPastDue': float(row.amount_past_due) if row.amount_past_due else 0,
                        'accountType': row.account_type,
                        'paymentRating': row.payment_rating,
                        'accountStatus': row.account_status
                    })
            
            user_data['credit'] = {
                'creditReports': [{
                    'creditReportData': {
                        'score': {'bureauScore': str(credit_score)},
                        'creditAccount': {
                            'creditAccountDetails': credit_accounts
                        },
                        'totalOutstanding': total_outstanding,
                        'totalPastDue': total_past_due
                    }
                }]
            }
            
            print(f"✅ Loaded credit data: Score {credit_score}, {len(credit_accounts)} accounts")
            
        except Exception as e:
            print(f"❌ Error loading credit: {e}")
            user_data['credit'] = {}
            self.validation_errors.append(f"Failed to load credit: {str(e)}")
        
        # 3. Load MF Transactions Data
        try:
            print(f"📊 Loading MF transactions for {self.master_uid}...")
            
            mf_query = f"""
                SELECT 
                    isin_number,
                    scheme_name,
                    transaction_type,
                    transaction_amount,
                    transaction_units,
                    purchase_price,
                    folio_id
                FROM `{self.project_id}.{self.dataset_id}.mf_transactions_processed`
                WHERE master_uid = @master_uid
                ORDER BY transaction_date DESC
            """
            
            query_job = self.client.query(mf_query, job_config=job_config)
            results = query_job.result()
            
            # Aggregate by scheme
            schemes = {}
            for row in results:
                isin = row.isin_number
                if isin not in schemes:
                    schemes[isin] = {
                        'scheme_name': row.scheme_name,
                        'total_invested': 0,
                        'total_units': 0,
                        'transactions': []
                    }
                
                amount = float(row.transaction_amount) if row.transaction_amount else 0
                units = float(row.transaction_units) if row.transaction_units else 0
                
                if row.transaction_type in ['PURCHASE', 'SIP']:
                    schemes[isin]['total_invested'] += amount
                    schemes[isin]['total_units'] += units
                elif row.transaction_type in ['REDEMPTION', 'SELL']:
                    schemes[isin]['total_invested'] -= amount
                    schemes[isin]['total_units'] -= units
                
                schemes[isin]['transactions'].append({
                    'type': row.transaction_type,
                    'amount': amount,
                    'units': units,
                    'price': float(row.purchase_price) if row.purchase_price else 0
                })
            
            user_data['mf_transactions'] = {
                'schemes': schemes,
                'total_schemes': len(schemes)
            }
            
            print(f"✅ Loaded MF data: {len(schemes)} schemes")
            
        except Exception as e:
            print(f"❌ Error loading MF transactions: {e}")
            user_data['mf_transactions'] = {}
            self.validation_errors.append(f"Failed to load MF transactions: {str(e)}")
        
        # 4. Load EPF Holdings Data
        try:
            print(f"📊 Loading EPF holdings for {self.master_uid}...")
            
            epf_query = f"""
                SELECT 
                    establishment_name,
                    member_id,
                    pf_balance,
                    employee_share,
                    employer_share,
                    pension_balance,
                    total_pf_balance
                FROM `{self.project_id}.{self.dataset_id}.epf_holdings_processed`
                WHERE master_uid = @master_uid
                ORDER BY processed_at DESC
                LIMIT 1
            """
            
            query_job = self.client.query(epf_query, job_config=job_config)
            results = query_job.result()
            
            epf_data = {}
            for row in results:
                epf_data = {
                    'establishment_name': row.establishment_name,
                    'member_id': row.member_id,
                    'pf_balance': float(row.pf_balance) if row.pf_balance else 0,
                    'employee_share': float(row.employee_share) if row.employee_share else 0,
                    'employer_share': float(row.employer_share) if row.employer_share else 0,
                    'pension_balance': float(row.pension_balance) if row.pension_balance else 0,
                    'total_pf_balance': float(row.total_pf_balance) if row.total_pf_balance else 0
                }
                break
            
            user_data['epf'] = epf_data
            
            total_epf = epf_data.get('total_pf_balance', 0)
            print(f"✅ Loaded EPF data: ₹{total_epf:,.2f}")
            
        except Exception as e:
            print(f"❌ Error loading EPF: {e}")
            user_data['epf'] = {}
            self.validation_errors.append(f"Failed to load EPF: {str(e)}")
        
        return user_data

    def load_user_data_from_local_files(self, file_paths: Optional[Dict[str, str]] = None) -> Dict:
        """Load user data from local JSON files (fallback)"""
        
        # Default file paths
        default_paths = {
            'net_worth': os.path.join(self.local_data_path, 'fetch_net_worth.json'),
            'credit': os.path.join(self.local_data_path, 'fetch_credit_report.json'),
            'epf': os.path.join(self.local_data_path, 'fetch_epf_details.json'),
            'mf_transactions': os.path.join(self.local_data_path, 'fetch_mf_transactions.json')
        }
        
        # Use provided paths or defaults
        data_sources = file_paths or default_paths
        user_data = {}
        
        print(f"📁 Loading data from local files in {self.local_data_path}...")
        
        for data_type, file_path in data_sources.items():
            try:
                if os.path.exists(file_path):
                    print(f"📊 Loading {data_type} from {file_path}...")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        user_data[data_type] = json.load(f)
                    print(f"✅ Loaded {data_type} data successfully")
                else:
                    print(f"⚠️ File not found: {file_path}")
                    user_data[data_type] = {}
                    self.missing_data_fields.append(f"local_file_{data_type}")
                    
            except Exception as e:
                print(f"❌ Error loading {data_type} from {file_path}: {e}")
                user_data[data_type] = {}
                self.validation_errors.append(f"Failed to load local {data_type}: {str(e)}")
        
        return user_data

    def load_user_data(self, file_paths: Optional[Dict[str, str]] = None) -> Dict:
        """Main data loading method with fallback support"""
        
        if self.use_local_files:
            # Primary: Load from local files
            user_data = self.load_user_data_from_local_files(file_paths)
        else:
            try:
                # Primary: Load from BigQuery
                user_data = self.load_user_data_from_bigquery()
            except Exception as e:
                print(f"⚠️ BigQuery loading failed: {e}")
                print("🔄 Falling back to local files...")
                self.use_local_files = True
                user_data = self.load_user_data_from_local_files(file_paths)
        
        # Validate data quality
        total_loaded = sum(1 for data in user_data.values() if data)
        print(f"📈 Data Quality: {total_loaded}/4 data sources loaded")
        
        self.data = user_data
        return user_data

    def extract_assets_robust(self) -> dict:
        """Extract assets from structured data"""
        assets = {}
        
        try:
            # From structured net worth data
            net_worth_data = self.data.get('net_worth', {})
            if 'assets' in net_worth_data:
                assets.update(net_worth_data['assets'])
            
            # Add account balances as bank deposits
            if 'accounts' in net_worth_data:
                bank_total = 0
                for account_id, account_info in net_worth_data['accounts'].items():
                    bank_total += account_info.get('balance', 0)
                
                if bank_total > 0:
                    assets['BANK_DEPOSITS'] = assets.get('BANK_DEPOSITS', 0) + bank_total
            
            # Add EPF as retirement asset
            epf_data = self.data.get('epf', {})
            if epf_data.get('total_pf_balance', 0) > 0:
                assets['EPF_BALANCE'] = epf_data['total_pf_balance']
            
            # Add MF investments
            mf_data = self.data.get('mf_transactions', {})
            if 'schemes' in mf_data:
                mf_total = sum(scheme.get('total_invested', 0) for scheme in mf_data['schemes'].values())
                if mf_total > 0:
                    assets['MUTUAL_FUNDS'] = mf_total
            
            # Apply differential privacy
            for key, value in assets.items():
                assets[key] = self.add_differential_privacy_noise(value)
            
            # Ensure minimum assets to avoid division by zero
            if not assets or sum(assets.values()) == 0:
                self.missing_data_fields.append('primary_assets')
                assets = {'ESTIMATED_ASSETS': 1000.0}
                
        except Exception as e:
            self.logger.error(f"Asset extraction failed: {e}")
            assets = {'ESTIMATED_ASSETS': 1000.0}
        
        return assets

    def extract_liabilities_robust(self) -> dict:
        """Extract liabilities from structured data"""
        liabilities = {}
        
        try:
            # From structured net worth data
            net_worth_data = self.data.get('net_worth', {})
            if 'liabilities' in net_worth_data:
                liabilities.update(net_worth_data['liabilities'])
            
            # From credit report data
            credit_data = self.data.get('credit', {})
            if 'creditReports' in credit_data and credit_data['creditReports']:
                try:
                    credit_report = credit_data['creditReports'][0]['creditReportData']
                    
                    # Calculate total credit debt
                    credit_debt = 0
                    if 'creditAccount' in credit_report:
                        accounts = credit_report['creditAccount'].get('creditAccountDetails', [])
                        for account in accounts:
                            current_balance = float(account.get('currentBalance', 0))
                            past_due = float(account.get('amountPastDue', 0))
                            credit_debt += current_balance + past_due
                    
                    if credit_debt > 0:
                        liabilities['CREDIT_DEBT'] = credit_debt
                        
                except (IndexError, KeyError):
                    self.missing_data_fields.append('credit_liabilities')
            
            # Apply differential privacy
            for key, value in liabilities.items():
                liabilities[key] = self.add_differential_privacy_noise(value)
            
        except Exception as e:
            self.logger.error(f"Liability extraction failed: {e}")
        
        return liabilities

    def _validate_credit_score(self, raw_score) -> int:
        """Validate credit score range"""
        try:
            score = int(float(str(raw_score)))
            return max(300, min(900, score))
        except (ValueError, TypeError):
            return 650

    def calculate_financial_ratios_safe(self, assets: dict, liabilities: dict) -> dict:
        """Calculate financial ratios with edge case handling"""
        total_assets = max(sum(assets.values()), 1)
        total_liabilities = sum(liabilities.values())
        
        # Credit analysis from structured data
        credit_score = 650
        credit_utilization = 0.0
        
        try:
            credit_data = self.data.get('credit', {})
            if 'creditReports' in credit_data and credit_data['creditReports']:
                credit_report = credit_data['creditReports'][0]['creditReportData']
                
                # Credit score
                raw_score = credit_report.get('score', {}).get('bureauScore', '650')
                credit_score = self._validate_credit_score(raw_score)
                
                # Credit utilization
                accounts = credit_report.get('creditAccount', {}).get('creditAccountDetails', [])
                total_outstanding = 0
                total_limit = 0
                
                for account in accounts:
                    outstanding = float(account.get('currentBalance', 0))
                    # Estimate credit limit (you might need to add this column to your schema)
                    estimated_limit = outstanding * 2  # Simple estimation
                    
                    total_outstanding += outstanding
                    total_limit += estimated_limit
                
                credit_utilization = total_outstanding / max(total_limit, 1)
                
        except Exception as e:
            self.logger.error(f"Credit analysis failed: {e}")
            self.missing_data_fields.append('credit_analysis')
        
        ratios = {
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'net_worth': total_assets - total_liabilities,
            'debt_to_asset_ratio': total_liabilities / total_assets,
            'credit_score': credit_score,
            'credit_utilization': min(credit_utilization, 1.0),
            'liquidity_ratio': assets.get('BANK_DEPOSITS', 0) / max(total_liabilities, 1),
            'investment_ratio': (assets.get('MUTUAL_FUNDS', 0) + assets.get('EPF_BALANCE', 0)) / total_assets
        }
        
        self.financial_ratios = ratios
        return ratios

    def analyze_investment_performance_robust(self) -> dict:
        """Analyze investment performance from structured MF data"""
        performance = {
            'total_invested': 0,
            'current_value': 0,
            'total_returns': 0,
            'schemes': [],
            'data_quality_issues': []
        }
        
        try:
            mf_data = self.data.get('mf_transactions', {})
            if 'schemes' in mf_data:
                schemes = mf_data['schemes']
                
                for isin, scheme_data in schemes.items():
                    invested = scheme_data.get('total_invested', 0)
                    units = scheme_data.get('total_units', 0)
                    
                    # Estimate current value (you might want to add current NAV to your data)
                    # For now, using a simple estimation
                    estimated_current = invested * 1.08  # Assuming 8% growth
                    
                    scheme_performance = {
                        'name': scheme_data.get('scheme_name', 'Unknown Scheme'),
                        'isin': isin,
                        'invested': invested,
                        'current': estimated_current,
                        'returns': estimated_current - invested,
                        'return_pct': ((estimated_current - invested) / invested * 100) if invested > 0 else 0,
                        'units': units
                    }
                    
                    performance['schemes'].append(scheme_performance)
                    performance['total_invested'] += invested
                    performance['current_value'] += estimated_current
                
                performance['total_returns'] = performance['current_value'] - performance['total_invested']
                
        except Exception as e:
            self.logger.error(f"Investment analysis failed: {e}")
            performance['data_quality_issues'].append(str(e))
        
        return performance

    # ... [Keep all the other methods exactly the same: generate_multi_timeframe_predictions, 
    # advanced_ml_model, calculate_data_quality_score, analyze_with_error_handling, 
    # store_results_in_bigquery, run_comprehensive_analysis, _display_results, etc.]

    def generate_multi_timeframe_predictions(self, timeframes: list = [3, 6, 9, 12, 24]) -> dict:
        """Generate rule-based predictions"""
        ratios = self.financial_ratios
        
        scenarios = {
            'Conservative': {'monthly_return': 0.005, 'savings_rate': 0.8},
            'Moderate': {'monthly_return': 0.007, 'savings_rate': 1.0},
            'Aggressive': {'monthly_return': 0.010, 'savings_rate': 1.2}
        }
        
        base_monthly_savings = max(ratios['total_assets'] * 0.02 / 12, 5000)
        monthly_loan_payment = min(ratios['total_liabilities'] * 0.05, 25000)
        
        all_predictions = {}
        
        for timeframe in timeframes:
            timeframe_predictions = {}
            
            for scenario_name, params in scenarios.items():
                projected_assets = ratios['total_assets']
                projected_liabilities = ratios['total_liabilities']
                monthly_savings = base_monthly_savings * params['savings_rate']
                
                for month in range(timeframe):
                    projected_assets += monthly_savings
                    projected_assets *= (1 + params['monthly_return'])
                    projected_liabilities = max(0, projected_liabilities - monthly_loan_payment)
                
                net_worth = projected_assets - projected_liabilities
                growth_pct = ((net_worth - ratios['net_worth']) / ratios['net_worth']) * 100
                
                timeframe_predictions[scenario_name] = {
                    'net_worth': round(net_worth, 2),
                    'growth_percentage': round(growth_pct, 1),
                    'projected_assets': round(projected_assets, 2),
                    'projected_liabilities': round(projected_liabilities, 2)
                }
            
            all_predictions[f'{timeframe}_months'] = timeframe_predictions
        
        return all_predictions

    def advanced_ml_model(self) -> dict:
        """ML model for predictions"""
        np.random.seed(42)
        n_samples = 500
        
        # Generate training data
        features = []
        targets = []
        
        for _ in range(n_samples):
            age = np.random.normal(35, 10)
            income = np.random.lognormal(11, 0.5)
            debt_ratio = np.random.beta(2, 5)
            credit_score = np.random.normal(700, 100)
            investment_return = np.random.normal(0.08, 0.03)
            savings_rate = np.random.beta(3, 7)
            
            base_assets = income * (age - 22) * 0.3
            debt_penalty = 1 - (debt_ratio * 0.5)
            credit_bonus = 1 + ((credit_score - 500) / 1000)
            investment_bonus = 1 + investment_return
            
            net_worth = base_assets * debt_penalty * credit_bonus * investment_bonus
            
            features.append([
                debt_ratio, credit_score, base_assets, age, income * savings_rate,
                investment_return, savings_rate, income, age * income / 100000
            ])
            targets.append(net_worth)
        
        # Train model
        X = np.array(features)
        y = np.array(targets)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        pred = model.predict(X_test)
        
        # User predictions
        ratios = self.financial_ratios
        estimated_income = ratios['total_assets'] * 0.15
        
        user_features = np.array([[
            ratios['debt_to_asset_ratio'],
            ratios['credit_score'],
            ratios['total_assets'],
            35,
            estimated_income * 0.2,
            0.08,
            0.2,
            estimated_income,
            35 * estimated_income / 100000
        ]])
        
        ml_predictions = {}
        for months in [3, 6, 9, 12, 24]:
            adjusted_features = user_features.copy()
            growth_factor = (1 + 0.007) ** months
            adjusted_features[0][2] *= growth_factor
            adjusted_features[0][0] *= (0.95 ** (months/12))
            
            pred = model.predict(adjusted_features)[0]
            ml_predictions[f'{months}_months'] = round(pred, 2)
        
        return {
            'predictions': ml_predictions,
            'model_performance': {
                'mae': mean_absolute_error(y_test, pred),
                'r2': r2_score(y_test, pred)
            }
        }

    def calculate_data_quality_score(self) -> float:
        """Calculate data quality score"""
        score = 100.0
        
        deductions = {
            'primary_assets': 15,
            'credit_data': 10,
            'investment_data': 10,
            'bank_accounts': 5,
            'transaction_history': 5
        }
        
        for field in self.missing_data_fields:
            for key, deduction in deductions.items():
                if key in field:
                    score -= deduction
                    break
        
        score -= len(self.validation_errors) * 5
        
        if len(self.data) == 4 and all(self.data.values()):
            score += 10
        
        return max(0.0, min(100.0, score))

    def analyze_with_error_handling(self) -> dict:
        """Main analysis with comprehensive error handling"""
        analysis_result = {
            'success': False,
            'data': {},
            'errors': [],
            'warnings': [],
            'data_quality_score': 0
        }
        
        try:
            # Extract data with fallbacks
            assets = self.extract_assets_robust()
            liabilities = self.extract_liabilities_robust()
            
            # Calculate ratios
            ratios = self.calculate_financial_ratios_safe(assets, liabilities)
            
            # Investment analysis
            investment_perf = self.analyze_investment_performance_robust()
            
            # Predictions
            rule_based_predictions = self.generate_multi_timeframe_predictions()
            ml_results = self.advanced_ml_model()
            
            # Calculate data quality score
            data_quality_score = self.calculate_data_quality_score()
            
            analysis_result.update({
                'success': True,
                'data': {
                    'assets': assets,
                    'liabilities': liabilities,
                    'ratios': ratios,
                    'investment_performance': investment_perf,
                    'rule_based_predictions': rule_based_predictions,
                    'ml_predictions': ml_results['predictions'],
                    'ml_model_performance': ml_results['model_performance']
                },
                'data_quality_score': data_quality_score,
                'missing_fields': self.missing_data_fields,
                'errors': self.validation_errors
            })
            
        except Exception as e:
            self.logger.error(f"Analysis failed for user {self.master_uid}: {str(e)}")
            analysis_result['errors'].append(f"Critical error: {str(e)}")
        
        return analysis_result

    def store_results_in_bigquery(self, analysis_result: dict) -> str:
        """Store net worth results in BigQuery with correct table names"""
        
        if not self.client:
            print("⚠️ BigQuery client not available, skipping storage")
            return None
        
        # Updated table names for net worth
        results_table_id = f"{self.project_id}.{self.dataset_id}.networth_results"
        webhook_table_id = f"{self.project_id}.{self.dataset_id}.networth_webhook"
        
        # Prepare data for insertion
        row_data = {
            'master_uid': self.master_uid,
            'analysis_id': self.analysis_id,
            'analysis_timestamp': datetime.utcnow(),
            'net_worth': analysis_result['data']['ratios']['net_worth'],
            'total_assets': analysis_result['data']['ratios']['total_assets'],
            'total_liabilities': analysis_result['data']['ratios']['total_liabilities'],
            'credit_score': int(analysis_result['data']['ratios']['credit_score']),
            'debt_to_asset_ratio': analysis_result['data']['ratios']['debt_to_asset_ratio'],
            'credit_utilization': analysis_result['data']['ratios']['credit_utilization'],
            'investment_ratio': analysis_result['data']['ratios']['investment_ratio'],
            'data_quality_score': analysis_result['data_quality_score'],
            'investment_performance': json.dumps(analysis_result['data']['investment_performance'], default=str),
            'rule_based_predictions': json.dumps(analysis_result['data']['rule_based_predictions'], default=str),
            'ml_predictions': json.dumps(analysis_result['data']['ml_predictions'], default=str),
            'validation_errors': json.dumps(self.validation_errors),
            'missing_data_fields': json.dumps(self.missing_data_fields),
            'raw_analysis_data': json.dumps({
                'assets': analysis_result['data']['assets'],
                'liabilities': analysis_result['data']['liabilities'],
                'ratios': analysis_result['data']['ratios']
            }, default=str)
        }
        
        try:
            # Insert into main results table
            errors = self.client.insert_rows_json(results_table_id, [row_data])
            if errors:
                self.logger.error(f"BigQuery insert failed: {errors}")
                return None
            
            # Update webhook cache table
            merge_query = f"""
                MERGE `{webhook_table_id}` T
                USING (
                    SELECT 
                        @master_uid as master_uid,
                        @analysis_id as latest_analysis_id,
                        @net_worth as latest_net_worth,
                        @credit_score as latest_credit_score,
                        @data_quality as latest_data_quality
                ) S
                ON T.master_uid = S.master_uid
                WHEN MATCHED THEN
                    UPDATE SET 
                        latest_analysis_id = S.latest_analysis_id,
                        latest_net_worth = S.latest_net_worth,
                        latest_credit_score = S.latest_credit_score,
                        latest_data_quality = S.latest_data_quality,
                        last_updated = CURRENT_TIMESTAMP()
                WHEN NOT MATCHED THEN
                    INSERT (master_uid, latest_analysis_id, latest_net_worth, latest_credit_score, latest_data_quality, last_updated)
                    VALUES (S.master_uid, S.latest_analysis_id, S.latest_net_worth, S.latest_credit_score, S.latest_data_quality, CURRENT_TIMESTAMP())
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("master_uid", "STRING", self.master_uid),
                    bigquery.ScalarQueryParameter("analysis_id", "STRING", self.analysis_id),
                    bigquery.ScalarQueryParameter("net_worth", "FLOAT64", row_data['net_worth']),
                    bigquery.ScalarQueryParameter("credit_score", "INT64", row_data['credit_score']),
                    bigquery.ScalarQueryParameter("data_quality", "FLOAT64", row_data['data_quality_score'])
                ]
            )
            
            query_job = self.client.query(merge_query, job_config=job_config)
            query_job.result()
            
            print(f"✅ Results stored successfully for {self.master_uid}")
            return self.analysis_id
            
        except Exception as e:
            self.logger.error(f"Failed to store results: {e}")
            return None

    def run_comprehensive_analysis(self, file_paths: Optional[Dict[str, str]] = None):
        """Main analysis function"""
        print("🚀 Starting Fiscal Fox Net Worth Analysis (Structured Tables)...")
        print(f"📁 Data Source: {'Local Files' if self.use_local_files else 'BigQuery Structured Tables'}")
        
        # Load data
        print(f"\n📥 Loading data for {self.master_uid}...")
        user_data = self.load_user_data(file_paths)
        
        if not any(user_data.values()):
            print("❌ No data found. Please check your data sources.")
            return None
        
        # Run analysis
        print("\n🔬 Running comprehensive net worth analysis...")
        analysis_result = self.analyze_with_error_handling()
        
        if not analysis_result['success']:
            print(f"❌ Analysis failed: {analysis_result['errors']}")
            return None
        
        # Display results
        self._display_results(analysis_result)
        
        # Store in BigQuery (if available)
        if not self.use_local_files:
            print(f"\n💾 Storing results in BigQuery...")
            result_id = self.store_results_in_bigquery(analysis_result)
            
            if result_id:
                print(f"✅ Net Worth Analysis completed! Result ID: {result_id}")
        else:
            print(f"\n💾 Saving results to local file...")
            self._save_results_locally(analysis_result)
        
        return analysis_result

    def _save_results_locally(self, analysis_result: dict):
        """Save analysis results to local JSON file"""
        try:
            results_file = os.path.join(self.local_data_path, f"analysis_results_{self.analysis_id}.json")
            
            # Add metadata
            analysis_result['metadata'] = {
                'analysis_id': self.analysis_id,
                'timestamp': datetime.utcnow().isoformat(),
                'master_uid': self.master_uid
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, default=str)
            
            print(f"✅ Results saved to: {results_file}")
            
        except Exception as e:
            print(f"❌ Failed to save results locally: {e}")

    def _display_results(self, results):
        """Display comprehensive results"""
        print("\n" + "=" * 70)
        print("🦊 FISCAL FOX NET WORTH ANALYSIS RESULTS (Structured Data)")
        print("=" * 70)
        
        ratios = results['data']['ratios']
        print(f"\n💰 CURRENT NET WORTH POSITION:")
        print(f"• Net Worth: ₹{ratios['net_worth']:,.2f}")
        print(f"• Total Assets: ₹{ratios['total_assets']:,.2f}")
        print(f"• Total Liabilities: ₹{ratios['total_liabilities']:,.2f}")
        print(f"• Data Quality Score: {results['data_quality_score']:.1f}/100")
        
        print(f"\n📊 KEY FINANCIAL RATIOS:")
        print(f"• Debt-to-Asset Ratio: {ratios['debt_to_asset_ratio']:.2%}")
        print(f"• Credit Score: {ratios['credit_score']:.0f}")
        print(f"• Credit Utilization: {ratios['credit_utilization']:.2%}")
        print(f"• Investment Ratio: {ratios['investment_ratio']:.2%}")
        
        # Investment performance
        investment_perf = results['data']['investment_performance']
        if investment_perf['total_invested'] > 0:
            roi = (investment_perf['total_returns'] / investment_perf['total_invested']) * 100
            print(f"\n📈 INVESTMENT PERFORMANCE:")
            print(f"• Total Invested: ₹{investment_perf['total_invested']:,.2f}")
            print(f"• Current Value: ₹{investment_perf['current_value']:,.2f}")
            print(f"• Total Returns: ₹{investment_perf['total_returns']:,.2f}")
            print(f"• Overall ROI: {roi:.2f}%")
        
        # Rule-based predictions
        print(f"\n🎯 RULE-BASED NET WORTH PREDICTIONS:")
        for timeframe, scenarios in results['data']['rule_based_predictions'].items():
            print(f"\n{timeframe.replace('_', ' ').title()}:")
            for scenario, values in scenarios.items():
                print(f"  • {scenario}: ₹{values['net_worth']:,.0f} ({values['growth_percentage']:+.1f}%)")
        
        # ML predictions
        print(f"\n🤖 ML NET WORTH PREDICTIONS:")
        for period, prediction in results['data']['ml_predictions'].items():
            print(f"• {period.replace('_', ' ').title()}: ₹{prediction:,.2f}")
        
        # Data quality issues
        if self.validation_errors or self.missing_data_fields:
            print(f"\n⚠️ DATA QUALITY ISSUES:")
            for error in self.validation_errors:
                print(f"• {error}")
            if self.missing_data_fields:
                print(f"• Missing data fields: {len(self.missing_data_fields)}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_latest_networth_results(master_uid: str = "ff_user_8a838f3528819407") -> dict:
    """Webhook function to get latest net worth results"""
    
    try:
        client = bigquery.Client(project="fiscal-fox-fin")
        
        query = f"""
            SELECT * FROM `fiscal-fox-fin.fiscal_master_dw.networth_webhook`
            WHERE master_uid = @master_uid
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("master_uid", "STRING", master_uid)
            ]
        )
        
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()
        
        for row in results:
            return {
                'master_uid': row.master_uid,
                'latest_analysis_id': row.latest_analysis_id,
                'latest_net_worth': row.latest_net_worth,
                'latest_credit_score': row.latest_credit_score,
                'latest_data_quality': row.latest_data_quality,
                'last_updated': row.last_updated.isoformat()
            }
        
        return {'error': 'No net worth analysis found for this master_uid'}
        
    except Exception as e:
        return {'error': f'Failed to fetch net worth results: {str(e)}'}


def create_sample_data_files(data_path: str = "data/"):
    """Create sample data files for testing"""
    os.makedirs(data_path, exist_ok=True)
    
    # Sample net worth data (structured format)
    net_worth_sample = {
        "assets": {
            "BANK_DEPOSITS": 500000,
            "MUTUAL_FUNDS": 300000
        },
        "liabilities": {
            "CREDIT_CARD": 50000
        },
        "accounts": {
            "acc_001": {
                "account_type": "SAVINGS",
                "bank_name": "HDFC Bank",
                "balance": 250000
            }
        },
        "total_net_worth": 750000
    }
    
    # Sample credit data
    credit_sample = {
        "creditReports": [{
            "creditReportData": {
                "score": {"bureauScore": "750"},
                "creditAccount": {
                    "creditAccountDetails": [
                        {
                            "currentBalance": 25000,
                            "amountPastDue": 0,
                            "accountType": "Credit Card"
                        }
                    ]
                }
            }
        }]
    }
    
    # Save sample files
    with open(os.path.join(data_path, "fetch_net_worth.json"), "w") as f:
        json.dump(net_worth_sample, f, indent=2)
    
    with open(os.path.join(data_path, "fetch_credit_report.json"), "w") as f:
        json.dump(credit_sample, f, indent=2)
    
    # Create empty files for missing data
    for filename in ["fetch_epf_details.json", "fetch_mf_transactions.json"]:
        with open(os.path.join(data_path, filename), "w") as f:
            json.dump({}, f)
    
    print(f"✅ Sample data files created in {data_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Fiscal Fox Net Worth Analyzer - Structured Tables')
    parser.add_argument('--master_uid', default="ff_user_8a838f3528819407", help='Master UID')
    parser.add_argument('--use_local_files', action='store_true', help='Use local files instead of BigQuery')
    parser.add_argument('--data_path', default="data/", help='Path to local data files')
    parser.add_argument('--create_sample_data', action='store_true', help='Create sample data files')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        create_sample_data_files(args.data_path)
        return
    
    # Standard Analysis Mode
    print("🦊 Starting Fiscal Fox Net Worth Analyzer (Structured Tables)...")
    
    analyzer = FiscalFoxNetWorthAnalyzer(
        master_uid=args.master_uid,
        use_local_files=args.use_local_files,
        local_data_path=args.data_path
    )
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        print(f"\n🔗 Analysis completed successfully!")
        print(f"📊 Net Worth: ₹{results['data']['ratios']['net_worth']:,.2f}")
        print(f"🎯 Data Quality: {results['data_quality_score']:.1f}/100")
        
        if not args.use_local_files:
            print(f"\n🌐 To get results via webhook, call:")
            print(f"get_latest_networth_results('{args.master_uid}')")


if __name__ == "__main__":
    main()

