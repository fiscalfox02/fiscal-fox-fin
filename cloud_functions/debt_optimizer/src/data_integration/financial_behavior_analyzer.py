"""
Financial Behavior Analyzer - Extract behavioral patterns from financial data
Analyzes user behavior patterns across spending, investing, and debt management
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd

class FinancialBehaviorAnalyzer:
    def __init__(self):
        self.behavior_categories = {
            'spending_discipline': 0.3,
            'investment_consistency': 0.25,
            'debt_management': 0.25,
            'financial_planning': 0.2
        }
        
        self.behavior_thresholds = {
            'excellent': 85,
            'good': 70,
            'fair': 55,
            'poor': 40,
            'critical': 0
        }
        
        # Behavioral indicators
        self.discipline_indicators = {
            'regular_savings': 10,
            'consistent_payments': 15,
            'budget_adherence': 10,
            'emergency_fund': 5
        }
    
    def analyze_complete_behavior_profile(self, financial_data: Dict) -> Dict:
        """Generate complete behavioral profile from all financial data"""
        print("Analyzing complete financial behavior profile...")
        
        behavior_profile = {
            'overall_score': 0,
            'behavior_category': '',
            'individual_scores': {},
            'behavioral_insights': [],
            'behavior_trends': {},
            'risk_profile': {},
            'recommendations': []
        }
        
        try:
            # Analyze each behavior category
            spending_analysis = self.analyze_spending_discipline(
                financial_data.get('bank_transactions', {})
            )
            behavior_profile['individual_scores']['spending_discipline'] = spending_analysis
            
            investment_analysis = self.analyze_investment_behavior(
                financial_data.get('stock_transactions', {})
            )
            behavior_profile['individual_scores']['investment_consistency'] = investment_analysis
            
            debt_analysis = self.analyze_debt_behavior(
                financial_data.get('credit_report', {}),
                financial_data.get('bank_transactions', {})
            )
            behavior_profile['individual_scores']['debt_management'] = debt_analysis
            
            planning_analysis = self.analyze_financial_planning(financial_data)
            behavior_profile['individual_scores']['financial_planning'] = planning_analysis
            
            # Calculate overall behavior score
            behavior_profile['overall_score'] = self.calculate_behavior_score(
                behavior_profile['individual_scores']
            )
            
            # Determine behavior category
            behavior_profile['behavior_category'] = self._get_behavior_category(
                behavior_profile['overall_score']
            )
            
            # Generate insights and recommendations
            behavior_profile['behavioral_insights'] = self.generate_behavior_insights(
                behavior_profile['individual_scores']
            )
            
            behavior_profile['risk_profile'] = self._assess_behavioral_risk_profile(
                behavior_profile['individual_scores']
            )
            
            print(f"Behavior analysis completed - Overall Score: {behavior_profile['overall_score']:.1f}")
            return behavior_profile
            
        except Exception as e:
            print(f" Error in behavior analysis: {e}")
            return self._create_empty_behavior_profile()
    
    def analyze_spending_discipline(self, bank_transactions: Dict) -> Dict:
        """Analyze spending discipline patterns from bank transaction data"""
        
        discipline_score = 0
        max_score = 100
        
        analysis = {
            'score': 0,
            'category': 'spending_discipline',
            'indicators': {},
            'patterns': {},
            'strengths': [],
            'weaknesses': []
        }
        
        # Get transaction analysis data
        spending_patterns = bank_transactions.get('spending_patterns', {})
        cash_flow_analysis = bank_transactions.get('cash_flow_analysis', {})
        
        if not spending_patterns and not cash_flow_analysis:
            return analysis
        
        # 1. Budget adherence analysis
        budget_score = self._analyze_budget_adherence(spending_patterns, cash_flow_analysis)
        discipline_score += budget_score
        analysis['indicators']['budget_adherence'] = budget_score
        
        # 2. Discretionary spending control
        discretionary_score = self._analyze_discretionary_spending(spending_patterns)
        discipline_score += discretionary_score
        analysis['indicators']['discretionary_control'] = discretionary_score
        
        # 3. Spending consistency
        consistency_score = self._analyze_spending_consistency(cash_flow_analysis)
        discipline_score += consistency_score
        analysis['indicators']['spending_consistency'] = consistency_score
        
        # 4. Emergency expense management
        emergency_score = self._analyze_emergency_expenses(spending_patterns)
        discipline_score += emergency_score
        analysis['indicators']['emergency_management'] = emergency_score
        
        # Generate patterns and insights
        analysis['patterns'] = self._extract_spending_patterns(bank_transactions)
        analysis['score'] = min(discipline_score, max_score)
        
        # Strengths and weaknesses
        if analysis['score'] >= 70:
            analysis['strengths'].append("Excellent spending discipline")
        if discretionary_score < 20:
            analysis['weaknesses'].append("High discretionary spending")
        
        return analysis
    
    def analyze_investment_behavior(self, stock_transactions: Dict) -> Dict:
        """Analyze investment behavior consistency from stock transaction data"""
        
        investment_score = 0
        max_score = 100
        
        analysis = {
            'score': 0,
            'category': 'investment_consistency',
            'indicators': {},
            'patterns': {},
            'strengths': [],
            'weaknesses': []
        }
        
        portfolio_analysis = stock_transactions.get('portfolio_analysis', {})
        investment_behavior = stock_transactions.get('investment_behavior', {})
        
        if not portfolio_analysis:
            return analysis
        
        # 1. Investment frequency and consistency
        frequency_score = self._analyze_investment_frequency(stock_transactions)
        investment_score += frequency_score
        analysis['indicators']['investment_frequency'] = frequency_score
        
        # 2. Diversification behavior
        diversification_score = self._analyze_diversification_behavior(portfolio_analysis)
        investment_score += diversification_score
        analysis['indicators']['diversification'] = diversification_score
        
        # 3. Long-term vs short-term behavior
        horizon_score = self._analyze_investment_horizon(stock_transactions)
        investment_score += horizon_score
        analysis['indicators']['investment_horizon'] = horizon_score
        
        # 4. Performance consistency
        performance_score = self._analyze_investment_performance_consistency(portfolio_analysis)
        investment_score += performance_score
        analysis['indicators']['performance_consistency'] = performance_score
        
        analysis['score'] = min(investment_score, max_score)
        analysis['patterns'] = self._extract_investment_patterns(stock_transactions)
        
        # Generate insights
        if analysis['score'] >= 75:
            analysis['strengths'].append("Consistent investment approach")
        if diversification_score < 15:
            analysis['weaknesses'].append("Limited portfolio diversification")
        
        return analysis
    
    def analyze_debt_behavior(self, debt_data: Dict, bank_transactions: Dict) -> Dict:
        """Analyze debt management behavior from credit report and payment patterns"""
        
        debt_score = 0
        max_score = 100
        
        analysis = {
            'score': 0,
            'category': 'debt_management',
            'indicators': {},
            'patterns': {},
            'strengths': [],
            'weaknesses': []
        }
        
        extracted_debts = debt_data.get('extracted_debts', [])
        debt_payment_behavior = bank_transactions.get('debt_payment_behavior', {})
        
        # 1. Payment timeliness
        timeliness_score = self._analyze_payment_timeliness(extracted_debts, debt_payment_behavior)
        debt_score += timeliness_score
        analysis['indicators']['payment_timeliness'] = timeliness_score
        
        # 2. Debt utilization behavior
        utilization_score = self._analyze_debt_utilization(extracted_debts)
        debt_score += utilization_score
        analysis['indicators']['debt_utilization'] = utilization_score
        
        # 3. Debt reduction progress
        reduction_score = self._analyze_debt_reduction_behavior(debt_payment_behavior)
        debt_score += reduction_score
        analysis['indicators']['debt_reduction'] = reduction_score
        
        # 4. Credit mix management
        mix_score = self._analyze_credit_mix_behavior(extracted_debts)
        debt_score += mix_score
        analysis['indicators']['credit_mix'] = mix_score
        
        analysis['score'] = min(debt_score, max_score)
        analysis['patterns'] = self._extract_debt_patterns(debt_data, bank_transactions)
        
        # Generate insights
        if len([d for d in extracted_debts if d.get('past_due', 0) == 0]) == len(extracted_debts):
            analysis['strengths'].append("No past due payments")
        if any(d.get('past_due', 0) > 0 for d in extracted_debts):
            analysis['weaknesses'].append("Past due payments detected")
        
        return analysis
    
    def analyze_financial_planning(self, complete_data: Dict) -> Dict:
        """Analyze financial planning behavior from complete financial picture"""
        
        planning_score = 0
        max_score = 100
        
        analysis = {
            'score': 0,
            'category': 'financial_planning',
            'indicators': {},
            'patterns': {},
            'strengths': [],
            'weaknesses': []
        }
        
        # 1. Emergency fund planning
        emergency_score = self._analyze_emergency_fund_planning(complete_data)
        planning_score += emergency_score
        analysis['indicators']['emergency_fund_planning'] = emergency_score
        
        # 2. Investment portfolio planning
        portfolio_score = self._analyze_portfolio_planning(complete_data)
        planning_score += portfolio_score
        analysis['indicators']['portfolio_planning'] = portfolio_score
        
        # 3. Debt management planning
        debt_planning_score = self._analyze_debt_planning(complete_data)
        planning_score += debt_planning_score
        analysis['indicators']['debt_planning'] = debt_planning_score
        
        # 4. Cash flow planning
        cashflow_score = self._analyze_cashflow_planning(complete_data)
        planning_score += cashflow_score
        analysis['indicators']['cashflow_planning'] = cashflow_score
        
        analysis['score'] = min(planning_score, max_score)
        
        return analysis
    
    def generate_behavior_insights(self, behavior_data: Dict) -> List[str]:
        """Generate actionable behavioral insights from analysis"""
        
        insights = []
        
        # Spending discipline insights
        spending = behavior_data.get('spending_discipline', {})
        if spending.get('score', 0) >= 75:
            insights.append("You demonstrate excellent spending discipline with controlled discretionary expenses")
        elif spending.get('score', 0) < 50:
            insights.append("Consider implementing budgeting strategies to improve spending discipline")
        
        # Investment behavior insights
        investment = behavior_data.get('investment_consistency', {})
        if investment.get('score', 0) >= 70:
            insights.append("Your investment approach shows good consistency and diversification")
        elif investment.get('score', 0) < 50:
            insights.append("Consider developing a more systematic investment strategy")
        
        # Debt management insights
        debt = behavior_data.get('debt_management', {})
        if debt.get('score', 0) >= 80:
            insights.append("You manage debt payments excellently with no past due amounts")
        elif debt.get('score', 0) < 60:
            insights.append("Focus on improving debt payment consistency and reducing utilization")
        
        # Financial planning insights
        planning = behavior_data.get('financial_planning', {})
        if planning.get('score', 0) >= 75:
            insights.append("You show strong financial planning with adequate emergency funds")
        elif planning.get('score', 0) < 50:
            insights.append("Develop a comprehensive financial plan including emergency fund building")
        
        return insights[:4]  # Return top 4 insights
    
    def calculate_behavior_score(self, behavior_analysis: Dict) -> float:
        """Calculate overall behavior score from individual categories"""
        
        total_weighted_score = 0
        total_weight = 0
        
        for category, weight in self.behavior_categories.items():
            category_data = behavior_analysis.get(category, {})
            category_score = category_data.get('score', 0)
            
            total_weighted_score += category_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return total_weighted_score / total_weight
    
    # Helper methods for specific analysis
    def _analyze_budget_adherence(self, spending_patterns: Dict, cash_flow_analysis: Dict) -> float:
        """Analyze budget adherence behavior"""
        
        summary_metrics = cash_flow_analysis.get('summary_metrics', {})
        avg_surplus = summary_metrics.get('avg_monthly_surplus', 0)
        avg_income = summary_metrics.get('avg_monthly_income', 1)
        
        if avg_income <= 0:
            return 0
        
        surplus_ratio = avg_surplus / avg_income
        
        # Score based on surplus ratio
        if surplus_ratio >= 0.2:  # Saving 20%+
            return 30
        elif surplus_ratio >= 0.1:  # Saving 10%+
            return 20
        elif surplus_ratio >= 0.05:  # Saving 5%+
            return 15
        elif surplus_ratio >= 0:  # Breaking even
            return 10
        else:  # Deficit
            return 0
    
    def _analyze_discretionary_spending(self, spending_patterns: Dict) -> float:
        """Analyze discretionary spending control"""
        
        by_essential_type = spending_patterns.get('by_essential_type', {})
        
        total_spending = sum(cat.get('amount', 0) for cat in by_essential_type.values())
        discretionary_spending = by_essential_type.get('discretionary', {}).get('amount', 0)
        
        if total_spending <= 0:
            return 0
        
        discretionary_ratio = discretionary_spending / total_spending
        
        # Score based on discretionary ratio (lower is better)
        if discretionary_ratio <= 0.15:  # Less than 15%
            return 25
        elif discretionary_ratio <= 0.25:  # Less than 25%
            return 20
        elif discretionary_ratio <= 0.35:  # Less than 35%
            return 15
        else:  # High discretionary spending
            return 5
    
    def _analyze_investment_frequency(self, stock_transactions: Dict) -> float:
        """Analyze investment frequency and consistency"""
        
        raw_transactions = stock_transactions.get('raw_transactions', [])
        
        if not raw_transactions:
            return 0
        
        # Count buy transactions
        buy_transactions = [t for t in raw_transactions if t.get('transaction_type') == 1]
        
        if len(buy_transactions) >= 12:  # Monthly or more frequent
            return 25
        elif len(buy_transactions) >= 6:  # Bi-monthly
            return 20
        elif len(buy_transactions) >= 3:  # Quarterly
            return 15
        else:  # Infrequent
            return 5
    
    def _analyze_diversification_behavior(self, portfolio_analysis: Dict) -> float:
        """Analyze portfolio diversification behavior"""
        
        securities_count = portfolio_analysis.get('portfolio_securities', 0)
        
        if securities_count >= 10:
            return 25
        elif securities_count >= 7:
            return 20
        elif securities_count >= 5:
            return 15
        elif securities_count >= 3:
            return 10
        else:
            return 0
    
    def _get_behavior_category(self, overall_score: float) -> str:
        """Get behavior category based on overall score"""
        
        if overall_score >= self.behavior_thresholds['excellent']:
            return "Excellent Financial Discipline"
        elif overall_score >= self.behavior_thresholds['good']:
            return "Good Financial Management"
        elif overall_score >= self.behavior_thresholds['fair']:
            return "Fair Financial Habits"
        elif overall_score >= self.behavior_thresholds['poor']:
            return "Poor Financial Discipline"
        else:
            return "Critical Financial Behavior Issues"
    
    def _assess_behavioral_risk_profile(self, individual_scores: Dict) -> Dict:
        """Assess behavioral risk profile"""
        
        risk_factors = []
        
        # Check for high-risk behaviors
        spending_score = individual_scores.get('spending_discipline', {}).get('score', 0)
        if spending_score < 50:
            risk_factors.append("Poor spending discipline")
        
        debt_score = individual_scores.get('debt_management', {}).get('score', 0)
        if debt_score < 60:
            risk_factors.append("Inconsistent debt management")
        
        investment_score = individual_scores.get('investment_consistency', {}).get('score', 0)
        if investment_score < 40:
            risk_factors.append("Lack of investment planning")
        
        # Determine overall risk level
        total_risk_factors = len(risk_factors)
        if total_risk_factors == 0:
            risk_level = "Low"
        elif total_risk_factors <= 2:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'risk_score': total_risk_factors * 25  # 0-100 scale
        }
    
    def _create_empty_behavior_profile(self) -> Dict:
        """Create empty behavior profile for error cases"""
        return {
            'overall_score': 0,
            'behavior_category': 'Insufficient Data',
            'individual_scores': {},
            'behavioral_insights': ['Insufficient data for behavior analysis'],
            'behavior_trends': {},
            'risk_profile': {'risk_level': 'Unknown', 'risk_factors': [], 'risk_score': 0},
            'recommendations': ['Provide more transaction data for comprehensive analysis']
        }
    
    # Additional helper methods (abbreviated for space)
    def _analyze_spending_consistency(self, cash_flow_analysis: Dict) -> float:
        return 15  # Placeholder implementation
    
    def _analyze_emergency_expenses(self, spending_patterns: Dict) -> float:
        return 10  # Placeholder implementation
    
    def _extract_spending_patterns(self, bank_transactions: Dict) -> Dict:
        return {}  # Placeholder implementation
    
    def _analyze_investment_horizon(self, stock_transactions: Dict) -> float:
        return 20  # Placeholder implementation
    
    def _analyze_investment_performance_consistency(self, portfolio_analysis: Dict) -> float:
        return 15  # Placeholder implementation
    
    def _extract_investment_patterns(self, stock_transactions: Dict) -> Dict:
        return {}  # Placeholder implementation
    
    def _analyze_payment_timeliness(self, extracted_debts: List[Dict], debt_payment_behavior: Dict) -> float:
        past_due_count = sum(1 for debt in extracted_debts if debt.get('past_due', 0) > 0)
        return max(0, 30 - (past_due_count * 10))
    
    def _analyze_debt_utilization(self, extracted_debts: List[Dict]) -> float:
        return 20  # Placeholder implementation
    
    def _analyze_debt_reduction_behavior(self, debt_payment_behavior: Dict) -> float:
        return 15  # Placeholder implementation
    
    def _analyze_credit_mix_behavior(self, extracted_debts: List[Dict]) -> float:
        return 10  # Placeholder implementation
    
    def _extract_debt_patterns(self, debt_data: Dict, bank_transactions: Dict) -> Dict:
        return {}  # Placeholder implementation
    
    def _analyze_emergency_fund_planning(self, complete_data: Dict) -> float:
        return 25  # Placeholder implementation
    
    def _analyze_portfolio_planning(self, complete_data: Dict) -> float:
        return 25  # Placeholder implementation
    
    def _analyze_debt_planning(self, complete_data: Dict) -> float:
        return 25  # Placeholder implementation
    
    def _analyze_cashflow_planning(self, complete_data: Dict) -> float:
        return 25  # Placeholder implementation

# Test function
def test_financial_behavior_analyzer():
    """Test the financial behavior analyzer"""
    print(" Testing Financial Behavior Analyzer...")
    
    # Sample financial data
    sample_data = {
        'bank_transactions': {
            'spending_patterns': {
                'by_essential_type': {
                    'discretionary': {'amount': 15000},
                    'essential': {'amount': 35000}
                }
            },
            'cash_flow_analysis': {
                'summary_metrics': {
                    'avg_monthly_income': 80000,
                    'avg_monthly_surplus': 15000
                }
            }
        },
        'stock_transactions': {
            'portfolio_analysis': {
                'portfolio_securities': 8,
                'current_portfolio_value': 500000
            },
            'raw_transactions': [{'transaction_type': 1} for _ in range(12)]
        },
        'credit_report': {
            'extracted_debts': [
                {'past_due': 0, 'credit_limit': 100000, 'current_balance': 25000}
            ]
        }
    }
    
    analyzer = FinancialBehaviorAnalyzer()
    results = analyzer.analyze_complete_behavior_profile(sample_data)
    
    print(f" Behavior Analysis Results:")
    print(f"Overall Score: {results['overall_score']:.1f}/100")
    print(f"Category: {results['behavior_category']}")
    print(f"Risk Level: {results['risk_profile']['risk_level']}")
    
    return results

if __name__ == "__main__":
    test_financial_behavior_analyzer()
