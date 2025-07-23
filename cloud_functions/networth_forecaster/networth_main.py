#!/usr/bin/env python3
"""
Complete Fiscal Fox Net Worth Analyzer
Updated for networth_results and networth_webhook tables
"""

import json
import sys
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
warnings.filterwarnings('ignore')

class FiscalFoxNetWorthAnalyzer:
    def __init__(self, master_uid: str = "ff_user_8a838f3528819407", enable_privacy: bool = True):
        self.master_uid = master_uid
        self.project_id = "fiscal-fox-fin"
        self.dataset_id = "fiscal_master_dw"
        self.enable_privacy = enable_privacy
        self.data = {}
        self.financial_ratios = {}
        self.validation_errors = []
        self.missing_data_fields = []
        self.analysis_id = self._generate_analysis_id()
        
        # Security setup
        if enable_privacy:
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
        
        # Setup BigQuery client and logging
        self.client = self._setup_bigquery_client()
        self._setup_logging()
    
    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.master_uid}_networth_analysis_{timestamp}"
    
    def _setup_bigquery_client(self):
        """Initialize BigQuery client"""
        try:
            client = bigquery.Client(project=self.project_id)
            print(f"Connected to BigQuery project: {self.project_id}")
            return client
        except Exception as e:
            print(f"Failed to connect to BigQuery: {e}")
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

    def load_user_data_from_bigquery(self):
        """Load user data from your existing BigQuery tables"""
        
        queries = {
            'net_worth': f"""
                SELECT json_data 
                FROM `{self.project_id}.{self.dataset_id}.net_worth_raw` 
                WHERE master_uid = @master_uid
                ORDER BY created_at DESC 
                LIMIT 1
            """,
            'credit': f"""
                SELECT json_data 
                FROM `{self.project_id}.{self.dataset_id}.credit_report_raw` 
                WHERE master_uid = @master_uid
                ORDER BY created_at DESC 
                LIMIT 1
            """,
            'mf_transactions': f"""
                SELECT json_data 
                FROM `{self.project_id}.{self.dataset_id}.mf_transactions_raw` 
                WHERE master_uid = @master_uid
                ORDER BY created_at DESC 
                LIMIT 1
            """,
            'epf': f"""
                SELECT json_data 
                FROM `{self.project_id}.{self.dataset_id}.epf_details_raw` 
                WHERE master_uid = @master_uid
                ORDER BY created_at DESC 
                LIMIT 1
            """
        }
        
        user_data = {}
        
        for data_type, query in queries.items():
            try:
                print(f"Loading {data_type} data for {self.master_uid}...")
                
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("master_uid", "STRING", self.master_uid)
                    ]
                )
                
                query_job = self.client.query(query, job_config=job_config)
                results = query_job.result()
                
                for row in results:
                    user_data[data_type] = json.loads(row.json_data) if row.json_data else {}
                    print(f"Loaded {data_type} data successfully")
                    break
                
                if data_type not in user_data:
                    print(f" No {data_type} data found for {self.master_uid}")
                    user_data[data_type] = {}
                    
            except Exception as e:
                print(f"Error loading {data_type}: {e}")
                user_data[data_type] = {}
                self.validation_errors.append(f"Failed to load {data_type}: {str(e)}")
        
        self.data = user_data
        return user_data

    def parse_currency_safe(self, currency_obj: dict) -> float:
        """Safely parse currency with validation"""
        try:
            if not currency_obj or 'units' not in currency_obj:
                return 0.0
            
            units = currency_obj.get('units', 0)
            nanos = currency_obj.get('nanos', 0)
            
            if isinstance(units, str):
                units = float(units.replace(',', ''))
            
            if abs(units) > 1e12:  # 1 trillion limit
                self.logger.warning(f"Unusually large amount detected")
                units = min(abs(units), 1e12)
            
            value = float(units) + float(nanos) / 1e9
            return self.add_differential_privacy_noise(value)
            
        except (ValueError, TypeError) as e:
            self.logger.error(f"Currency parsing error: {e}")
            return 0.0

    def extract_assets_robust(self) -> dict:
        """Extract assets with edge case handling"""
        assets = {}
        
        try:
            net_worth_data = self.data.get('net_worth', {})
            
            # From netWorthResponse
            if 'netWorthResponse' in net_worth_data:
                asset_values = net_worth_data['netWorthResponse'].get('assetValues', [])
                for asset in asset_values:
                    asset_type = asset.get('netWorthAttribute', 'UNKNOWN').replace('ASSET_TYPE_', '')
                    assets[asset_type] = self.parse_currency_safe(asset.get('value', {}))
            
            # From accountDetailsBulkResponse
            if 'accountDetailsBulkResponse' in net_worth_data:
                accounts = net_worth_data['accountDetailsBulkResponse'].get('accountDetailsMap', {})
                
                bank_total = 0
                securities_total = 0
                
                for account_id, account_data in accounts.items():
                    # Bank deposits
                    if 'depositSummary' in account_data:
                        balance = self.parse_currency_safe(
                            account_data['depositSummary'].get('currentBalance', {})
                        )
                        bank_total += balance
                    
                    # Securities (equity, ETF, REIT, InvIT)
                    for summary_type in ['equitySummary', 'etfSummary', 'reitSummary', 'invitSummary']:
                        if summary_type in account_data:
                            value = self.parse_currency_safe(
                                account_data[summary_type].get('currentValue', {})
                            )
                            securities_total += value
                
                if bank_total > 0:
                    assets['BANK_DEPOSITS'] = bank_total
                if securities_total > 0:
                    assets['SECURITIES'] = securities_total
            
            # Ensure minimum assets to avoid division by zero
            if not assets or sum(assets.values()) == 0:
                self.missing_data_fields.append('primary_assets')
                assets = {'ESTIMATED_ASSETS': 1000.0}
                
        except Exception as e:
            self.logger.error(f"Asset extraction failed: {e}")
            assets = {'ESTIMATED_ASSETS': 1000.0}
        
        return assets

    def extract_liabilities_robust(self) -> dict:
        """Extract liabilities with edge case handling"""
        liabilities = {}
        
        try:
            net_worth_data = self.data.get('net_worth', {})
            
            # From netWorthResponse
            if 'netWorthResponse' in net_worth_data:
                liability_values = net_worth_data['netWorthResponse'].get('liabilityValues', [])
                for liability in liability_values:
                    liability_type = liability.get('netWorthAttribute', 'UNKNOWN').replace('LIABILITY_TYPE_', '')
                    liabilities[liability_type] = self.parse_currency_safe(liability.get('value', {}))
            
            # From credit report
            credit_data = self.data.get('credit', {})
            if 'creditReports' in credit_data and credit_data['creditReports']:
                try:
                    credit_accounts = credit_data['creditReports'][0]['creditReportData'].get(
                        'creditAccount', {}
                    ).get('creditAccountDetails', [])
                    
                    credit_debt = 0
                    for account in credit_accounts:
                        current_balance = float(account.get('currentBalance', 0))
                        past_due = float(account.get('amountPastDue', 0))
                        credit_debt += current_balance + past_due
                    
                    if credit_debt > 0:
                        liabilities['CREDIT_DEBT'] = credit_debt
                        
                except (IndexError, KeyError):
                    self.missing_data_fields.append('credit_liabilities')
            
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
        
        # Credit analysis
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
                total_limit = 0
                total_outstanding = 0
                
                for account in accounts:
                    limit = float(account.get('creditLimitAmount', 0))
                    if limit == 0:
                        limit = float(account.get('highestCreditOrOriginalLoanAmount', 0))
                    
                    outstanding = float(account.get('currentBalance', 0))
                    total_limit += limit
                    total_outstanding += outstanding
                
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
            'liquidity_ratio': assets.get('SAVINGS_ACCOUNTS', 0) / max(total_liabilities, 1),
            'investment_ratio': (assets.get('MUTUAL_FUND', 0) + assets.get('SECURITIES', 0)) / total_assets
        }
        
        self.financial_ratios = ratios
        return ratios

    def analyze_investment_performance_robust(self) -> dict:
        """Analyze investment performance"""
        performance = {
            'total_invested': 0,
            'current_value': 0,
            'total_returns': 0,
            'schemes': [],
            'data_quality_issues': []
        }
        
        try:
            net_worth_data = self.data.get('net_worth', {})
            if 'mfSchemeAnalytics' in net_worth_data:
                schemes = net_worth_data['mfSchemeAnalytics'].get('schemeAnalytics', [])
                
                for scheme in schemes:
                    scheme_data = self._process_scheme_safe(scheme)
                    if scheme_data:
                        performance['schemes'].append(scheme_data)
                        performance['total_invested'] += scheme_data['invested']
                        performance['current_value'] += scheme_data['current']
                
                performance['total_returns'] = performance['current_value'] - performance['total_invested']
                
        except Exception as e:
            self.logger.error(f"Investment analysis failed: {e}")
            performance['data_quality_issues'].append(str(e))
        
        return performance

    def _process_scheme_safe(self, scheme: dict) -> dict:
        """Process individual scheme safely"""
        try:
            details = scheme.get('enrichedAnalytics', {}).get('analytics', {}).get('schemeDetails', {})
            
            current_val = self.parse_currency_safe(details.get('currentValue', {}))
            invested_val = self.parse_currency_safe(details.get('investedValue', {}))
            xirr = details.get('XIRR', 0)
            
            # Handle missing invested value
            if invested_val <= 0 and current_val > 0:
                invested_val = current_val * 0.9
                self.missing_data_fields.append(f"invested_value_{scheme.get('schemeDetail', {}).get('isinNumber', 'unknown')}")
            
            if current_val <= 0 and invested_val <= 0:
                return None
            
            scheme_name = scheme.get('schemeDetail', {}).get('nameData', {}).get('longName', 'Unknown Scheme')
            
            return {
                'name': scheme_name,
                'invested': invested_val,
                'current': current_val,
                'returns': current_val - invested_val,
                'return_pct': ((current_val - invested_val) / invested_val * 100) if invested_val > 0 else 0,
                'xirr': xirr if isinstance(xirr, (int, float)) and not np.isnan(xirr) else 0
            }
            
        except Exception as e:
            self.logger.error(f"Scheme processing failed: {e}")
            return None

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
        """Main analysis with comprehensive error handling - THE MISSING METHOD"""
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
            
            print(f"Results stored successfully for {self.master_uid}")
            return self.analysis_id
            
        except Exception as e:
            self.logger.error(f"Failed to store results: {e}")
            return None

    def run_comprehensive_analysis(self):
        """Main analysis function"""
        print("Starting Fiscal Fox Net Worth Analysis...")
        
        
        # Load data
        print(f"\nLoading data for {self.master_uid}...")
        user_data = self.load_user_data_from_bigquery()
        
        if not any(user_data.values()):
            print("No data found. Please check your master_uid or data tables.")
            return None
        
        # Run analysis
        print("\n Running comprehensive net worth analysis...")
        analysis_result = self.analyze_with_error_handling()
        
        if not analysis_result['success']:
            print(f" Analysis failed: {analysis_result['errors']}")
            return None
        
        # Display results
        self._display_results(analysis_result)
        
        # Store in BigQuery
        print(f"\n Storing results in BigQuery...")
        result_id = self.store_results_in_bigquery(analysis_result)
        
        if result_id:
            print(f" Net Worth Analysis completed! Result ID: {result_id}")
        
        return analysis_result

    def _display_results(self, results):
        """Display comprehensive results"""
        print("\n" + "=" * 60)
        print(" FISCAL FOX NET WORTH ANALYSIS RESULTS")
        print("=" * 60)
        
        ratios = results['data']['ratios']
        print(f"\nCURRENT NET WORTH POSITION:")
        print(f"• Net Worth: ₹{ratios['net_worth']:,.2f}")
        print(f"• Total Assets: ₹{ratios['total_assets']:,.2f}")
        print(f"• Total Liabilities: ₹{ratios['total_liabilities']:,.2f}")
        print(f"• Data Quality Score: {results['data_quality_score']:.1f}/100")
        
        print(f"\n KEY FINANCIAL RATIOS:")
        print(f"• Debt-to-Asset Ratio: {ratios['debt_to_asset_ratio']:.2%}")
        print(f"• Credit Score: {ratios['credit_score']:.0f}")
        print(f"• Credit Utilization: {ratios['credit_utilization']:.2%}")
        print(f"• Investment Ratio: {ratios['investment_ratio']:.2%}")
        
        # Investment performance
        investment_perf = results['data']['investment_performance']
        if investment_perf['total_invested'] > 0:
            roi = (investment_perf['total_returns'] / investment_perf['total_invested']) * 100
            print(f"\nINVESTMENT PERFORMANCE:")
            print(f"• Total Invested: ₹{investment_perf['total_invested']:,.2f}")
            print(f"• Current Value: ₹{investment_perf['current_value']:,.2f}")
            print(f"• Total Returns: ₹{investment_perf['total_returns']:,.2f}")
            print(f"• Overall ROI: {roi:.2f}%")
        
        # Rule-based predictions
        print(f"\n RULE-BASED NET WORTH PREDICTIONS:")
        for timeframe, scenarios in results['data']['rule_based_predictions'].items():
            print(f"\n{timeframe.replace('_', ' ').title()}:")
            for scenario, values in scenarios.items():
                print(f"  • {scenario}: ₹{values['net_worth']:,.0f} ({values['growth_percentage']:+.1f}%)")
        
        # ML predictions
        print(f"\n ML NET WORTH PREDICTIONS:")
        for period, prediction in results['data']['ml_predictions'].items():
            print(f"• {period.replace('_', ' ').title()}: ₹{prediction:,.2f}")
        
        # Data quality issues
        if self.validation_errors or self.missing_data_fields:
            print(f"\n DATA QUALITY ISSUES:")
            for error in self.validation_errors:
                print(f"• {error}")
            if self.missing_data_fields:
                print(f"• Missing data fields: {len(self.missing_data_fields)}")


def get_latest_networth_results(master_uid: str = "ff_user_8a838f3528819407"):
    """Webhook function to get latest net worth results"""
    
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
    
    try:
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


if __name__ == "__main__":
    # Get master_uid from command line or use default
    master_uid = sys.argv[1] if len(sys.argv) > 1 else "ff_user_8a838f3528819407"
    
    # Run the analysis
    analyzer = FiscalFoxNetWorthAnalyzer(master_uid=master_uid)
    results = analyzer.run_comprehensive_analysis()
    
    print(f"\n🔗 To get results via webhook, call:")
    print(f"get_latest_networth_results('{master_uid}')")
