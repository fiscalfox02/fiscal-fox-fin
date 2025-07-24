import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from google.cloud import bigquery
import os

warnings.filterwarnings('ignore')

class FiscalFoxDataLoader:
    def __init__(self, project_id='fiscal-fox-fin', dataset_id='fiscal_master_dw'):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.client = bigquery.Client(project=project_id)
    
    def load_comprehensive_financial_data(self, master_uid):
        """Main method to load all financial data from BigQuery"""
        
        queries = {
            'net_worth': f"""
                SELECT json_data, created_at
                FROM `{self.project_id}.{self.dataset_id}.net_worth_processed` 
                WHERE master_uid = @master_uid
                ORDER BY created_at DESC 
                LIMIT 1
            """,
            'credit': f"""
                SELECT json_data, created_at
                FROM `{self.project_id}.{self.dataset_id}.credit_report_processed` 
                WHERE master_uid = @master_uid
                ORDER BY created_at DESC 
                LIMIT 1
            """,
            'mf_transactions': f"""
                SELECT json_data, created_at
                FROM `{self.project_id}.{self.dataset_id}.mf_transactions_processed` 
                WHERE master_uid = @master_uid
                ORDER BY created_at DESC 
                LIMIT 1
            """,
            'epf': f"""
                SELECT json_data, created_at
                FROM `{self.project_id}.{self.dataset_id}.epf_details_processed` 
                WHERE master_uid = @master_uid
                ORDER BY created_at DESC 
                LIMIT 1
            """
        }
        
        financial_data = {}
        
        for data_type, query in queries.items():
            try:
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("master_uid", "STRING", master_uid)
                    ]
                )
                
                query_job = self.client.query(query, job_config=job_config)
                results = list(query_job.result())
                
                if results:
                    json_data = json.loads(results[0]['json_data'])
                    financial_data[data_type] = json_data
                    print(f"✅ Loaded {data_type} data (updated: {results[0]['created_at']})")
                else:
                    print(f"⚠️ No {data_type} data found")
                    financial_data[data_type] = {}
                    
            except Exception as e:
                print(f"❌ Error loading {data_type}: {e}")
                financial_data[data_type] = {}
        
        return financial_data

class ComprehensiveDebtOptimizer:
    def __init__(self, enable_detailed_analysis=True):
        self.enable_detailed_analysis = enable_detailed_analysis
        self.financial_data = {}
        self.debt_portfolio = []
        self.asset_portfolio = {}
        self.financial_ratios = {}
        self.optimization_results = {}
        self.net_worth_metrics = {}
        # Add BigQuery loader
        self.data_loader = FiscalFoxDataLoader()
        
    def load_all_financial_data(self, master_uid=None, file_paths=None):
        """Load financial data from BigQuery or local files"""
        if master_uid:
            # Load from BigQuery using master_uid
            print(f"Loading financial data for master_uid: {master_uid}")
            self.financial_data = self.data_loader.load_comprehensive_financial_data(master_uid)
            
        elif file_paths:
            # Fallback: Load from local files
            print("Loading from local files...")
            data_sources = {
                'net_worth': file_paths.get('net_worth', 'data/fetch_net_worth.json'),
                'credit': file_paths.get('credit', 'data/fetch_credit_report.json'),
                'epf': file_paths.get('epf', 'data/fetch_epf_details.json'),
                'mf_transactions': file_paths.get('mf_transactions', 'data/fetch_mf_transactions.json')
            }
            
            for source, file_path in data_sources.items():
                try:
                    with open(file_path, 'r') as f:
                        self.financial_data[source] = json.load(f)
                    print(f"✅ Loaded {source} data")
                except FileNotFoundError:
                    print(f"⚠️ {source} file not found: {file_path}")
                    self.financial_data[source] = {}
                except Exception as e:
                    print(f"❌ Error loading {source}: {e}")
                    self.financial_data[source] = {}
        else:
            raise ValueError("Either master_uid or file_paths must be provided")
        
        return self.financial_data
    
    def parse_currency_safe(self, currency_obj):
        """Safely parse currency values"""
        try:
            if not currency_obj or 'units' not in currency_obj:
                return 0.0
            
            units = currency_obj.get('units', 0)
            nanos = currency_obj.get('nanos', 0)
            
            if isinstance(units, str):
                units = float(units.replace(',', ''))
            
            return float(units) + float(nanos) / 1e9
        except (ValueError, TypeError):
            return 0.0
    
    def extract_comprehensive_assets(self):
        """Extract all assets from net worth data"""
        assets = {}
        
        try:
            net_worth_data = self.financial_data.get('net_worth', {})
            
            # From netWorthResponse
            if 'netWorthResponse' in net_worth_data:
                asset_values = net_worth_data['netWorthResponse'].get('assetValues', [])
                for asset in asset_values:
                    asset_type = asset.get('netWorthAttribute', 'UNKNOWN').replace('ASSET_TYPE_', '')
                    assets[asset_type] = self.parse_currency_safe(asset.get('value', {}))
            
            # From account details
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
                    
                    # Securities
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
            
            # Add EPF assets
            epf_data = self.financial_data.get('epf', {})
            if 'uanAccounts' in epf_data and epf_data['uanAccounts']:
                epf_balance = float(epf_data['uanAccounts'][0]['rawDetails']['overall_pf_balance']['current_pf_balance'])
                pension_balance = float(epf_data['uanAccounts'][0]['rawDetails']['overall_pf_balance']['pension_balance'])
                assets['EPF_BALANCE'] = epf_balance
                assets['PENSION_BALANCE'] = pension_balance
            
        except Exception as e:
            print(f"Error extracting assets: {e}")
        
        self.asset_portfolio = assets
        return assets
    
    def extract_comprehensive_debts(self):
        """Extract all debts from multiple sources"""
        debts = []
        
        try:
            # From net worth liabilities
            net_worth_data = self.financial_data.get('net_worth', {})
            if 'netWorthResponse' in net_worth_data:
                liability_values = net_worth_data['netWorthResponse'].get('liabilityValues', [])
                for liability in liability_values:
                    liability_type = liability.get('netWorthAttribute', 'UNKNOWN').replace('LIABILITY_TYPE_', '')
                    amount = self.parse_currency_safe(liability.get('value', {}))
                    
                    if amount > 0:
                        debts.append({
                            'debt_name': f"{liability_type} Loan",
                            'debt_type': 'Loan',
                            'total_outstanding': amount,
                            'current_balance': amount,
                            'past_due': 0,
                            'interest_rate': self._estimate_interest_rate(liability_type),
                            'min_payment': amount * 0.05,  # 5% estimated monthly payment
                            'credit_limit': 0,
                            'portfolio_type': 'I',  # Installment
                            'payment_rating': 1  # Assume good for net worth reported debts
                        })
            
            # From credit report (detailed debt information)
            credit_data = self.financial_data.get('credit', {})
            if 'creditReports' in credit_data and credit_data['creditReports']:
                credit_accounts = credit_data['creditReports'][0]['creditReportData'].get(
                    'creditAccount', {}
                ).get('creditAccountDetails', [])
                
                for account in credit_accounts:
                    balance = float(account.get('currentBalance', 0))
                    past_due = float(account.get('amountPastDue', 0))
                    total_outstanding = balance + past_due
                    
                    if total_outstanding > 0:
                        # Get credit limit
                        credit_limit = float(account.get('creditLimitAmount', 0))
                        if credit_limit == 0:
                            credit_limit = float(account.get('highestCreditOrOriginalLoanAmount', 0))
                        
                        # Calculate minimum payment
                        if account.get('portfolioType') == 'R':  # Revolving credit
                            min_payment = max(total_outstanding * 0.05, 500)
                        else:  # Installment
                            min_payment = total_outstanding * 0.08
                        
                        debts.append({
                            'debt_name': account.get('subscriberName', 'Unknown Lender'),
                            'debt_type': 'Credit Card' if account.get('portfolioType') == 'R' else 'Loan',
                            'total_outstanding': total_outstanding,
                            'current_balance': balance,
                            'past_due': past_due,
                            'interest_rate': float(account.get('rateOfInterest', 15.0)) / 100,
                            'min_payment': min_payment,
                            'credit_limit': credit_limit,
                            'portfolio_type': account.get('portfolioType', 'I'),
                            'payment_rating': int(account.get('paymentRating', 3)),
                            'account_status': account.get('accountStatus', ''),
                            'payment_history': account.get('paymentHistoryProfile', '')
                        })
            
        except Exception as e:
            print(f"Error extracting debts: {e}")
        
        self.debt_portfolio = debts
        return debts
    
    def _estimate_interest_rate(self, liability_type):
        """Estimate interest rates for different loan types"""
        rate_estimates = {
            'HOME_LOAN': 0.08,     # 8%
            'PERSONAL_LOAN': 0.14,  # 14%
            'AUTO_LOAN': 0.09,     # 9%
            'EDUCATION_LOAN': 0.10, # 10%
            'OTHER': 0.12          # 12%
        }
        return rate_estimates.get(liability_type, 0.12)
    
    def calculate_comprehensive_financial_ratios(self):
        """Calculate comprehensive financial ratios using all data"""
        assets = self.asset_portfolio
        debts = self.debt_portfolio
        
        total_assets = sum(assets.values())
        total_debt = sum(debt['total_outstanding'] for debt in debts)
        total_past_due = sum(debt['past_due'] for debt in debts)
        
        # Credit analysis
        credit_score = 650
        credit_utilization = 0.0
        
        try:
            credit_data = self.financial_data.get('credit', {})
            if 'creditReports' in credit_data and credit_data['creditReports']:
                credit_report = credit_data['creditReports'][0]['creditReportData']
                
                # Credit score
                raw_score = credit_report.get('score', {}).get('bureauScore', '650')
                credit_score = max(300, min(900, int(float(str(raw_score)))))
                
                # Credit utilization
                total_limit = 0
                total_outstanding = 0
                for debt in debts:
                    if debt['debt_type'] == 'Credit Card':
                        total_limit += debt['credit_limit']
                        total_outstanding += debt['current_balance']
                
                credit_utilization = total_outstanding / max(total_limit, 1)
        except:
            pass
        
        # Calculate financial strength metrics
        liquid_assets = assets.get('SAVINGS_ACCOUNTS', 0) + assets.get('BANK_DEPOSITS', 0)
        investment_assets = (assets.get('MUTUAL_FUND', 0) + 
                           assets.get('SECURITIES', 0) + 
                           assets.get('EPF_BALANCE', 0))
        
        # Monthly payment capacity
        estimated_monthly_income = total_assets * 0.01  # 1% of assets as monthly income estimate
        total_monthly_debt_payments = sum(debt['min_payment'] for debt in debts)
        debt_to_income_ratio = total_monthly_debt_payments / max(estimated_monthly_income, 1)
        
        ratios = {
            'total_assets': total_assets,
            'total_debt': total_debt,
            'total_past_due': total_past_due,
            'net_worth': total_assets - total_debt,
            'debt_to_asset_ratio': total_debt / max(total_assets, 1),
            'credit_score': credit_score,
            'credit_utilization': min(credit_utilization, 1.0),
            'liquidity_ratio': liquid_assets / max(total_debt, 1),
            'investment_ratio': investment_assets / max(total_assets, 1),
            'debt_to_income_ratio': debt_to_income_ratio,
            'liquid_assets': liquid_assets,
            'investment_assets': investment_assets,
            'estimated_monthly_income': estimated_monthly_income,
            'financial_strength_score': self._calculate_financial_strength_score(
                total_assets, total_debt, credit_score, credit_utilization, liquid_assets
            )
        }
        
        self.financial_ratios = ratios
        return ratios
    
    def _calculate_financial_strength_score(self, assets, debt, credit_score, utilization, liquidity):
        """Calculate overall financial strength score (0-100)"""
        # Net worth score (0-30 points)
        net_worth = assets - debt
        net_worth_score = min(30, max(0, (net_worth / 1000000) * 10))  # 10 points per 1M net worth
        
        # Credit score (0-25 points)
        credit_score_normalized = min(25, max(0, (credit_score - 300) / 600 * 25))
        
        # Credit utilization score (0-20 points)
        utilization_score = max(0, 20 - (utilization * 20))  # Lower utilization = higher score
        
        # Liquidity score (0-15 points)
        liquidity_score = min(15, (liquidity / max(debt, 1)) * 15)
        
        # Asset diversity score (0-10 points)
        diversity_score = 10  # Assume good diversity if they have multiple data sources
        
        return net_worth_score + credit_score_normalized + utilization_score + liquidity_score + diversity_score
    
    def calculate_advanced_priority_score(self, debt):
        """Advanced priority scoring using comprehensive financial data"""
        ratios = self.financial_ratios
        
        # Base avalanche and snowball scores
        max_balance = max(d['total_outstanding'] for d in self.debt_portfolio)
        min_balance = min(d['total_outstanding'] for d in self.debt_portfolio)
        max_interest = max(d['interest_rate'] for d in self.debt_portfolio)
        
        # Snowball component (psychological wins)
        snowball_score = (max_balance - debt['total_outstanding']) / max(max_balance - min_balance, 1)
        
        # Avalanche component (interest savings)
        avalanche_score = debt['interest_rate'] / max(max_interest, 0.01)
        
        # Financial strength adjustments
        strength_score = ratios['financial_strength_score']
        
        # Liquidity consideration
        liquidity_bonus = 0
        if ratios['liquidity_ratio'] > 1.0:  # High liquidity
            liquidity_bonus = 0.1  # Can afford aggressive debt payoff
        elif ratios['liquidity_ratio'] < 0.3:  # Low liquidity
            liquidity_bonus = -0.1  # Should be more conservative
        
        # Investment opportunity cost
        investment_opportunity_cost = 0
        if debt['interest_rate'] < 0.12 and ratios['investment_ratio'] < 0.3:
            # Low interest debt + low investment ratio = consider investing instead
            investment_opportunity_cost = -0.05
        
        # Credit utilization impact
        utilization_urgency = 0
        if debt['debt_type'] == 'Credit Card' and ratios['credit_utilization'] > 0.3:
            utilization_urgency = 0.2  # High priority for credit score improvement
        
        # Past due penalty (critical)
        past_due_penalty = 0.3 if debt['past_due'] > 0 else 0
        
        # Payment history penalty
        payment_history_penalty = debt['payment_rating'] * 0.02
        
        # Calculate weighted score based on financial strength
        if strength_score > 70:  # Strong financial position
            avalanche_weight = 0.5  # Focus on interest savings
            snowball_weight = 0.2
        elif strength_score > 40:  # Moderate position
            avalanche_weight = 0.4  # Balanced approach
            snowball_weight = 0.3
        else:  # Weak position
            avalanche_weight = 0.3  # More conservative, focus on quick wins
            snowball_weight = 0.4
        
        base_score = (avalanche_weight * avalanche_score) + (snowball_weight * snowball_score)
        
        final_score = (base_score + 
                      liquidity_bonus + 
                      investment_opportunity_cost + 
                      utilization_urgency + 
                      past_due_penalty + 
                      payment_history_penalty)
        
        return final_score, {
            'snowball_component': snowball_score,
            'avalanche_component': avalanche_score,
            'liquidity_bonus': liquidity_bonus,
            'investment_opportunity_cost': investment_opportunity_cost,
            'utilization_urgency': utilization_urgency,
            'past_due_penalty': past_due_penalty,
            'payment_history_penalty': payment_history_penalty,
            'financial_strength_score': strength_score
        }
    
    def optimize_debt_with_surplus_analysis(self, extra_payment_budget=None):
        """Optimize debt repayment considering available surplus"""
        if not self.debt_portfolio:
            return {"error": "No debts found to optimize"}
        
        ratios = self.financial_ratios
        
        # Calculate available surplus for debt repayment
        if extra_payment_budget is None:
            # Estimate surplus based on financial position
            monthly_income = ratios['estimated_monthly_income']
            current_debt_payments = sum(debt['min_payment'] for debt in self.debt_portfolio)
            estimated_expenses = monthly_income * 0.6  # 60% for living expenses
            
            available_surplus = monthly_income - current_debt_payments - estimated_expenses
            extra_payment_budget = max(0, available_surplus * 0.8)  # 80% of surplus for debt
        
        # Calculate priority scores
        debt_priorities = []
        for debt in self.debt_portfolio:
            score, components = self.calculate_advanced_priority_score(debt)
            debt_priorities.append({
                'debt': debt,
                'priority_score': score,
                'score_components': components,
                'monthly_payment': debt['min_payment'],
                'extra_payment': 0
            })
        
        # Sort by priority
        debt_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Allocate extra payment budget
        remaining_budget = extra_payment_budget
        for debt_info in debt_priorities:
            if remaining_budget <= 0:
                break
            
            debt = debt_info['debt']
            
            # Prioritize past due amounts first
            if debt['past_due'] > 0:
                past_due_allocation = min(remaining_budget, debt['past_due'])
                debt_info['extra_payment'] += past_due_allocation
                remaining_budget -= past_due_allocation
            
            # Then allocate remaining budget to highest priority debt
            if remaining_budget > 0 and debt_info == debt_priorities[0]:
                debt_info['extra_payment'] += remaining_budget
                remaining_budget = 0
            
            debt_info['monthly_payment'] = debt['min_payment'] + debt_info['extra_payment']
        
        # Calculate payoff timelines
        self._calculate_payoff_timelines(debt_priorities)
        
        # Generate comprehensive recommendations
        recommendations = self._generate_comprehensive_recommendations(debt_priorities, extra_payment_budget)
        
        self.optimization_results = {
            'debt_priorities': debt_priorities,
            'recommendations': recommendations,
            'financial_summary': {
                'total_debt': ratios['total_debt'],
                'net_worth': ratios['net_worth'],
                'financial_strength_score': ratios['financial_strength_score'],
                'available_surplus': extra_payment_budget,
                'total_monthly_payments': sum(d['monthly_payment'] for d in debt_priorities)
            },
            'asset_analysis': self._analyze_asset_vs_debt_strategy()
        }
        
        return self.optimization_results
    
    def _calculate_payoff_timelines(self, debt_priorities):
        """Calculate payoff timelines for each debt"""
        for debt_info in debt_priorities:
            debt = debt_info['debt']
            monthly_payment = debt_info['monthly_payment']
            balance = debt['total_outstanding']
            monthly_rate = debt['interest_rate'] / 12
            
            if monthly_rate > 0 and monthly_payment > balance * monthly_rate:
                months = -np.log(1 - (balance * monthly_rate) / monthly_payment) / np.log(1 + monthly_rate)
                total_interest = (monthly_payment * months) - balance
            else:
                months = balance / monthly_payment if monthly_payment > 0 else float('inf')
                total_interest = 0
            
            debt_info['months_to_payoff'] = round(months, 1)
            debt_info['total_interest_paid'] = round(total_interest, 2)
            debt_info['payoff_date'] = (datetime.now() + timedelta(days=30 * months)).strftime('%Y-%m-%d')
    
    def _generate_investment_debt_strategy(self):
        """Generate investment vs debt payoff strategy recommendations"""
        ratios = self.financial_ratios
        
        recommendations = []
        
        # High-interest debt analysis (>12% interest)
        high_interest_debts = [d for d in self.debt_portfolio if d['interest_rate'] > 0.12]
        high_interest_total = sum(d['total_outstanding'] for d in high_interest_debts)
        
        if high_interest_debts:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'DEBT_PAYOFF',
                'reasoning': f"Pay off ₹{high_interest_total:,.0f} in high-interest debt (>12%) before investing",
                'expected_return': f"Guaranteed {max(d['interest_rate'] for d in high_interest_debts):.1%} return by eliminating debt",
                'timeframe': 'Immediate'
            })
        
        # Medium-interest debt analysis (8-12% interest)
        medium_interest_debts = [d for d in self.debt_portfolio if 0.08 <= d['interest_rate'] <= 0.12]
        medium_interest_total = sum(d['total_outstanding'] for d in medium_interest_debts)
        
        if medium_interest_debts:
            # Compare with potential investment returns
            expected_investment_return = 0.12  # 12% expected from equity markets
            avg_debt_rate = np.mean([d['interest_rate'] for d in medium_interest_debts])
            
            if avg_debt_rate > expected_investment_return:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'DEBT_PAYOFF',
                    'reasoning': f"Medium-interest debt ({avg_debt_rate:.1%}) exceeds expected investment returns (12%)",
                    'expected_return': f"Guaranteed {avg_debt_rate:.1%} vs risky 12% from investments",
                    'timeframe': '6-12 months'
                })
            else:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'action': 'BALANCED_APPROACH',
                    'reasoning': f"Split approach: 60% debt payoff, 40% investments for medium-interest debt",
                    'expected_return': f"Debt elimination ({avg_debt_rate:.1%}) + wealth building (12% potential)",
                    'timeframe': '12-18 months'
                })
        
        # Low-interest debt analysis (<8% interest)
        low_interest_debts = [d for d in self.debt_portfolio if d['interest_rate'] < 0.08]
        low_interest_total = sum(d['total_outstanding'] for d in low_interest_debts)
        
        if low_interest_debts:
            # Consider tax benefits for home loans
            home_loans = [d for d in low_interest_debts if 'HOME' in d.get('debt_name', '').upper()]
            
            if home_loans:
                recommendations.append({
                    'priority': 'LOW',
                    'action': 'INVEST_FIRST',
                    'reasoning': f"Home loan interest ({np.mean([d['interest_rate'] for d in home_loans]):.1%}) has tax benefits. Invest surplus in equity",
                    'expected_return': f"Tax-adjusted debt cost ~{np.mean([d['interest_rate'] for d in home_loans]) * 0.7:.1%} vs 12% investment potential",
                    'timeframe': 'Long-term (5+ years)'
                })
            else:
                recommendations.append({
                    'priority': 'LOW',
                    'action': 'INVEST_FIRST',
                    'reasoning': f"Low-interest debt ({np.mean([d['interest_rate'] for d in low_interest_debts]):.1%}) allows focus on wealth building",
                    'expected_return': f"Investment potential (12%) exceeds debt cost ({np.mean([d['interest_rate'] for d in low_interest_debts]):.1%})",
                    'timeframe': 'Long-term (3+ years)'
                })
        
        # Emergency fund consideration
        emergency_fund_needed = ratios['estimated_monthly_income'] * 6
        current_liquid = ratios['liquid_assets']
        
        if current_liquid < emergency_fund_needed:
            recommendations.insert(0, {
                'priority': 'CRITICAL',
                'action': 'BUILD_EMERGENCY_FUND',
                'reasoning': f"Build emergency fund to ₹{emergency_fund_needed:,.0f} before aggressive debt payoff",
                'expected_return': 'Financial security and avoiding future debt',
                'timeframe': 'Immediate (3-6 months)'
            })
        
        # Asset reallocation recommendations
        if ratios['investment_ratio'] > 0.7 and high_interest_total > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'action': 'ASSET_REALLOCATION',
                'reasoning': f"Consider liquidating {min(high_interest_total / ratios['total_assets'] * 100, 30):.0f}% of investments to eliminate high-interest debt",
                'expected_return': f"Immediate guaranteed return by eliminating {max([d['interest_rate'] for d in high_interest_debts]):.1%} debt cost",
                'timeframe': 'Immediate'
            })
        
        return recommendations
    
    def _analyze_asset_vs_debt_strategy(self):
        """Analyze whether to pay debt or invest based on assets"""
        ratios = self.financial_ratios
        liquid_assets = ratios['liquid_assets']
        total_debt = ratios['total_debt']
        
        strategy_analysis = {
            'emergency_fund_status': 'adequate' if liquid_assets >= (ratios['estimated_monthly_income'] * 6) else 'insufficient',
            'debt_vs_investment_recommendation': [],
            'asset_reallocation_suggestions': [],
            'investment_vs_debt_recommendations': self._generate_investment_debt_strategy()
        }
        
        # Emergency fund analysis
        emergency_fund_needed = ratios['estimated_monthly_income'] * 6
        if liquid_assets < emergency_fund_needed:
            strategy_analysis['asset_reallocation_suggestions'].append(
                f"Build emergency fund: Need ₹{emergency_fund_needed - liquid_assets:,.0f} more"
            )
        
        # Extract key recommendations for backward compatibility
        investment_debt_recs = strategy_analysis['investment_vs_debt_recommendations']
        
        for rec in investment_debt_recs:
            if rec['action'] in ['DEBT_PAYOFF', 'INVEST_FIRST', 'BALANCED_APPROACH']:
                strategy_analysis['debt_vs_investment_recommendation'].append(rec['reasoning'])
            elif rec['action'] == 'ASSET_REALLOCATION':
                strategy_analysis['asset_reallocation_suggestions'].append(rec['reasoning'])
        
        return strategy_analysis
    
    def _generate_comprehensive_recommendations(self, debt_priorities, budget):
        """Generate detailed recommendations based on complete financial picture"""
        ratios = self.financial_ratios
        
        recommendations = {
            'executive_summary': [],
            'priority_ranking': [],
            'financial_strategy': [],
            'emergency_actions': [],
            'long_term_planning': [],
            'credit_score_impact': []
        }
        
        # Executive summary
        strength_score = ratios['financial_strength_score']
        if strength_score > 70:
            summary = f"**Strong Financial Position** (Score: {strength_score:.0f}/100): You can afford aggressive debt payoff strategies"
        elif strength_score > 40:
            summary = f"**Moderate Financial Position** (Score: {strength_score:.0f}/100): Balanced approach recommended"
        else:
            summary = f"**Building Financial Strength** (Score: {strength_score:.0f}/100): Focus on emergency fund and high-priority debts"
        
        recommendations['executive_summary'].append(summary)
        
        # Priority rankings with comprehensive reasoning
        for i, debt_info in enumerate(debt_priorities[:5]):  # Top 5 priorities
            debt = debt_info['debt']
            components = debt_info['score_components']
            
            reasons = []
            if components['past_due_penalty'] > 0:
                reasons.append("**URGENT**: Overdue amount")
            if components['utilization_urgency'] > 0:
                reasons.append("High credit utilization impact")
            if components['avalanche_component'] > 0.7:
                reasons.append(f"High interest rate ({debt['interest_rate']:.1%})")
            if components['investment_opportunity_cost'] < 0:
                reasons.append("Consider investment opportunity cost")
            
            reasoning = f"**#{i+1}: {debt['debt_name']}** - {', '.join(reasons) if reasons else 'Balanced priority factors'}"
            recommendations['priority_ranking'].append(reasoning)
        
        # Financial strategy based on net worth position
        net_worth = ratios['net_worth']
        if net_worth > 0:
            if ratios['liquidity_ratio'] > 1.0:
                strategy = f"**Asset-Rich Strategy**: With ₹{net_worth:,.0f} net worth and good liquidity, focus on interest optimization"
            else:
                strategy = f"**Conservative Strategy**: Positive net worth but low liquidity requires careful cash flow management"
        else:
            strategy = f"**Recovery Strategy**: Negative net worth of ₹{abs(net_worth):,.0f} requires aggressive debt reduction"
        
        recommendations['financial_strategy'].append(strategy)
        
        # Emergency actions
        past_due_debts = [d for d in debt_priorities if d['debt']['past_due'] > 0]
        if past_due_debts:
            emergency = f"**IMMEDIATE**: Address {len(past_due_debts)} overdue accounts totaling ₹{sum(d['debt']['past_due'] for d in past_due_debts):,.0f}"
            recommendations['emergency_actions'].append(emergency)
        
        # Credit score impact
        if ratios['credit_utilization'] > 0.3:
            credit_impact = f"**High Priority**: Reduce credit utilization from {ratios['credit_utilization']:.1%} to below 30% for significant credit score improvement"
            recommendations['credit_score_impact'].append(credit_impact)
        
        return recommendations
    
    def generate_comprehensive_report(self):
        """Generate the complete debt optimization report"""
        if not self.optimization_results:
            print("Please run optimize_debt_with_surplus_analysis() first")
            return
        
        print("=" * 80)
        print("🎯 COMPREHENSIVE DEBT OPTIMIZATION REPORT")
        print("=" * 80)
        
        # Financial Summary
        summary = self.optimization_results['financial_summary']
        print(f"\n💰 **FINANCIAL OVERVIEW:**")
        print(f"• Net Worth: ₹{summary['net_worth']:,.2f}")
        print(f"• Total Debt: ₹{summary['total_debt']:,.2f}")
        print(f"• Financial Strength Score: {summary['financial_strength_score']:.0f}/100")
        print(f"• Available Monthly Surplus: ₹{summary['available_surplus']:,.0f}")
        
        # Executive Summary
        print(f"\n📊 **EXECUTIVE SUMMARY:**")
        for summary_point in self.optimization_results['recommendations']['executive_summary']:
            print(f"• {summary_point}")
        
        # Priority Rankings
        print(f"\n🏆 **DEBT PRIORITY RANKING:**")
        for ranking in self.optimization_results['recommendations']['priority_ranking']:
            print(f"{ranking}")
        
        # Financial Strategy
        print(f"\n🧠 **FINANCIAL STRATEGY:**")
        for strategy in self.optimization_results['recommendations']['financial_strategy']:
            print(f"• {strategy}")
        
        # Emergency Actions
        if self.optimization_results['recommendations']['emergency_actions']:
            print(f"\n⚠️ **EMERGENCY ACTIONS:**")
            for action in self.optimization_results['recommendations']['emergency_actions']:
                print(f"• {action}")
        
        # Detailed Payment Plan
        print(f"\n💳 **OPTIMIZED PAYMENT PLAN:**")
        debt_priorities = self.optimization_results['debt_priorities']
        for debt_info in debt_priorities:
            debt = debt_info['debt']
            print(f"\n  📋 **{debt['debt_name']}:**")
            print(f"    • Type: {debt['debt_type']}")
            print(f"    • Outstanding: ₹{debt['total_outstanding']:,.0f}")
            print(f"    • Interest Rate: {debt['interest_rate']:.1%}")
            print(f"    • Monthly Payment: ₹{debt_info['monthly_payment']:,.0f}")
            if debt_info['extra_payment'] > 0:
                print(f"    • Extra Payment: ₹{debt_info['extra_payment']:,.0f}")
            print(f"    • Payoff Timeline: {debt_info['months_to_payoff']} months")
            print(f"    • Total Interest: ₹{debt_info['total_interest_paid']:,.0f}")
        
        # Asset vs Debt Analysis
        asset_analysis = self.optimization_results['asset_analysis']
        print(f"\n🏦 **ASSET VS DEBT ANALYSIS:**")
        print(f"• Emergency Fund: {asset_analysis['emergency_fund_status'].title()}")
        
        if asset_analysis['debt_vs_investment_recommendation']:
            print(f"• Investment Strategy:")
            for rec in asset_analysis['debt_vs_investment_recommendation']:
                print(f"  - {rec}")
        
        if asset_analysis['asset_reallocation_suggestions']:
            print(f"• Asset Reallocation:")
            for suggestion in asset_analysis['asset_reallocation_suggestions']:
                print(f"  - {suggestion}")
        
        # Credit Score Impact
        if self.optimization_results['recommendations']['credit_score_impact']:
            print(f"\n📈 **CREDIT SCORE IMPACT:**")
            for impact in self.optimization_results['recommendations']['credit_score_impact']:
                print(f"• {impact}")
        
        # Investment vs Debt Strategy
        if 'investment_vs_debt_recommendations' in asset_analysis:
            print(f"\n💡 **INVESTMENT VS DEBT STRATEGY:**")
            for rec in asset_analysis['investment_vs_debt_recommendations']:
                priority_emoji = {'CRITICAL': '🚨', 'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}
                emoji = priority_emoji.get(rec['priority'], '📋')
                print(f"\n  {emoji} **{rec['priority']} Priority - {rec['action'].replace('_', ' ').title()}:**")
                print(f"    • Strategy: {rec['reasoning']}")
                print(f"    • Expected Outcome: {rec['expected_return']}")
                print(f"    • Timeline: {rec['timeframe']}")

# Updated Usage Example
def run_comprehensive_debt_optimizer(master_uid=None, use_local_files=False):
    """Main function to run the comprehensive debt optimizer"""
    
    # Initialize optimizer
    optimizer = ComprehensiveDebtOptimizer()
    
    if master_uid and not use_local_files:
        # Load from BigQuery using master_uid
        print(f"Starting Fiscal Fox Net Worth Analysis for {master_uid}...")
        financial_data = optimizer.load_all_financial_data(master_uid=master_uid)
    else:
        # Fallback to local files
        file_paths = {
            'net_worth': 'data/fetch_net_worth.json',
            'credit': 'data/fetch_credit_report.json', 
            'epf': 'data/fetch_epf_details.json',
            'mf_transactions': 'data/fetch_mf_transactions.json'
        }
        financial_data = optimizer.load_all_financial_data(file_paths=file_paths)
    
    # Extract assets and debts
    assets = optimizer.extract_comprehensive_assets()
    debts = optimizer.extract_comprehensive_debts()
    
    print(f"Loaded {len(assets)} asset categories and {len(debts)} debt accounts")
    
    # Calculate comprehensive financial ratios
    ratios = optimizer.calculate_comprehensive_financial_ratios()
    
    # Run optimization with surplus analysis
    results = optimizer.optimize_debt_with_surplus_analysis()
    
    # Generate comprehensive report
    optimizer.generate_comprehensive_report()
    
    return optimizer

# Run the optimizer
if __name__ == "__main__":
    # Option 1: Use with BigQuery (production)
    optimizer = run_comprehensive_debt_optimizer(master_uid="ff_user_8a838f3528819407")
    
    # Option 2: Use with local files (testing)
    # optimizer = run_comprehensive_debt_optimizer(use_local_files=True)
