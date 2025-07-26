"""
Transaction Analyzer - Analyze real transaction patterns and insights
Processes stock and bank transactions to extract meaningful patterns
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import re

class TransactionAnalyzer:
    def __init__(self):
        self.analysis_categories = [
            'spending_patterns',
            'income_patterns', 
            'payment_behavior',
            'investment_activity',
            'cash_flow_trends',
            'transaction_frequency',
            'seasonal_patterns'
        ]
        
        # Spending categories with keywords
        self.spending_categories = {
            'food_delivery': {
                'keywords': ['swiggy', 'zomato', 'uber eats', 'dunzo', 'food'],
                'type': 'discretionary'
            },
            'transportation': {
                'keywords': ['uber', 'ola', 'metro', 'bus', 'taxi', 'petrol', 'fuel'],
                'type': 'semi_essential'
            },
            'utilities': {
                'keywords': ['electricity', 'water', 'gas', 'broadband', 'act', 'airtel'],
                'type': 'essential'
            },
            'financial_services': {
                'keywords': ['cred', 'credit card', 'loan', 'emi', 'mutual fund', 'investment'],
                'type': 'financial'
            },
            'shopping': {
                'keywords': ['amazon', 'flipkart', 'myntra', 'shopping', 'retail'],
                'type': 'discretionary'
            },
            'healthcare': {
                'keywords': ['hospital', 'medical', 'pharmacy', 'doctor', 'health'],
                'type': 'essential'
            },
            'entertainment': {
                'keywords': ['netflix', 'amazon prime', 'spotify', 'movie', 'entertainment'],
                'type': 'discretionary'
            }
        }
    
    def analyze_transaction_patterns(self, transactions: List[Dict]) -> Dict:
        """Analyze comprehensive transaction patterns"""
        print("Analyzing comprehensive transaction patterns...")
        
        if not transactions:
            return self._create_empty_analysis()
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(transactions)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['transaction_amount'] = df['transaction_amount'].astype(float)
        
        analysis_results = {
            'total_transactions': len(transactions),
            'date_range': {
                'start_date': df['transaction_date'].min().isoformat(),
                'end_date': df['transaction_date'].max().isoformat(),
                'days_covered': (df['transaction_date'].max() - df['transaction_date'].min()).days
            },
            'transaction_categorization': self.categorize_transactions(transactions),
            'spending_behavior': self.analyze_spending_behavior(transactions),
            'income_patterns': self.analyze_income_stability(transactions),
            'frequency_analysis': self.analyze_transaction_frequency(transactions),
            'seasonal_patterns': self.analyze_seasonal_patterns(df),
            'outlier_detection': self.detect_transaction_outliers(df),
            'trends': self.analyze_transaction_trends(df)
        }
        
        print(f" Analyzed {len(transactions)} transactions across {analysis_results['date_range']['days_covered']} days")
        return analysis_results
    
    def categorize_transactions(self, transactions: List[Dict]) -> Dict:
        """Categorize transactions by type and purpose"""
        
        categorized = {
            'by_type': defaultdict(lambda: {'count': 0, 'amount': 0}),
            'by_category': defaultdict(lambda: {'count': 0, 'amount': 0}),
            'by_essential_type': defaultdict(lambda: {'count': 0, 'amount': 0}),
            'uncategorized': []
        }
        
        for txn in transactions:
            amount = abs(float(txn['transaction_amount']))
            narration = txn.get('transaction_narration', '').lower()
            txn_type = txn.get('transaction_type', 0)
            
            # Categorize by transaction type (credit/debit)
            type_name = self._get_transaction_type_name(txn_type)
            categorized['by_type'][type_name]['count'] += 1
            categorized['by_type'][type_name]['amount'] += amount
            
            # Categorize by spending category
            category_found = False
            for category, data in self.spending_categories.items():
                if any(keyword in narration for keyword in data['keywords']):
                    categorized['by_category'][category]['count'] += 1
                    categorized['by_category'][category]['amount'] += amount
                    
                    # Also categorize by essential type
                    essential_type = data['type']
                    categorized['by_essential_type'][essential_type]['count'] += 1
                    categorized['by_essential_type'][essential_type]['amount'] += amount
                    
                    category_found = True
                    break
            
            if not category_found and txn_type == 2:  # Debit transactions
                categorized['uncategorized'].append({
                    'narration': txn.get('transaction_narration', ''),
                    'amount': amount,
                    'date': txn['transaction_date']
                })
        
        # Calculate percentages
        total_spending = sum(cat['amount'] for cat in categorized['by_category'].values())
        for category in categorized['by_category']:
            if total_spending > 0:
                categorized['by_category'][category]['percentage'] = (
                    categorized['by_category'][category]['amount'] / total_spending * 100
                )
        
        return dict(categorized)
    
    def analyze_spending_behavior(self, transactions: List[Dict]) -> Dict:
        """Analyze spending behavior patterns"""
        
        spending_analysis = {
            'total_spending': 0,
            'avg_transaction_amount': 0,
            'spending_frequency': 0,
            'high_value_transactions': [],
            'recurring_patterns': [],
            'spending_velocity': {},
            'discretionary_ratio': 0
        }
        
        debit_transactions = [t for t in transactions if t.get('transaction_type') == 2]
        if not debit_transactions:
            return spending_analysis
        
        amounts = [abs(float(t['transaction_amount'])) for t in debit_transactions]
        
        spending_analysis['total_spending'] = sum(amounts)
        spending_analysis['avg_transaction_amount'] = np.mean(amounts)
        spending_analysis['spending_frequency'] = len(debit_transactions)
        
        # High value transactions (top 10% by amount)
        high_value_threshold = np.percentile(amounts, 90)
        spending_analysis['high_value_transactions'] = [
            {
                'amount': abs(float(t['transaction_amount'])),
                'narration': t.get('transaction_narration', ''),
                'date': t['transaction_date']
            }
            for t in debit_transactions 
            if abs(float(t['transaction_amount'])) >= high_value_threshold
        ]
        
        # Analyze spending velocity (daily/weekly patterns)
        spending_analysis['spending_velocity'] = self._analyze_spending_velocity(debit_transactions)
        
        # Calculate discretionary spending ratio
        categorized = self.categorize_transactions(transactions)
        discretionary_spending = sum(
            categorized['by_essential_type'].get(cat, {}).get('amount', 0)
            for cat in ['discretionary']
        )
        
        if spending_analysis['total_spending'] > 0:
            spending_analysis['discretionary_ratio'] = discretionary_spending / spending_analysis['total_spending']
        
        return spending_analysis
    
    def analyze_income_stability(self, transactions: List[Dict]) -> Dict:
        """Analyze income patterns and stability"""
        
        income_analysis = {
            'total_income': 0,
            'avg_monthly_income': 0,
            'income_sources': [],
            'income_stability_score': 0,
            'salary_patterns': {},
            'irregular_income': []
        }
        
        credit_transactions = [t for t in transactions if t.get('transaction_type') == 1]
        if not credit_transactions:
            return income_analysis
        
        # Identify salary and other income sources
        salary_keywords = ['salary', 'sal', 'pay', 'wage', 'emp']
        interest_keywords = ['interest', 'int', 'saving']
        
        salary_transactions = []
        interest_transactions = []
        other_income = []
        
        for txn in credit_transactions:
            amount = float(txn['transaction_amount'])
            narration = txn.get('transaction_narration', '').lower()
            
            if any(keyword in narration for keyword in salary_keywords):
                salary_transactions.append(txn)
            elif any(keyword in narration for keyword in interest_keywords):
                interest_transactions.append(txn)
            else:
                other_income.append(txn)
        
        # Calculate income metrics
        income_analysis['total_income'] = sum(float(t['transaction_amount']) for t in credit_transactions)
        
        # Monthly income analysis
        if credit_transactions:
            df = pd.DataFrame(credit_transactions)
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            df['month'] = df['transaction_date'].dt.to_period('M')
            monthly_income = df.groupby('month')['transaction_amount'].sum()
            
            income_analysis['avg_monthly_income'] = monthly_income.mean()
            income_analysis['income_stability_score'] = self._calculate_income_stability(monthly_income)
        
        # Salary pattern analysis
        if salary_transactions:
            income_analysis['salary_patterns'] = self._analyze_salary_patterns(salary_transactions)
        
        income_analysis['income_sources'] = {
            'salary': len(salary_transactions),
            'interest': len(interest_transactions), 
            'other': len(other_income)
        }
        
        return income_analysis
    
    def analyze_transaction_frequency(self, transactions: List[Dict]) -> Dict:
        """Analyze transaction frequency patterns"""
        
        if not transactions:
            return {}
        
        df = pd.DataFrame(transactions)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['day_of_week'] = df['transaction_date'].dt.day_name()
        df['hour'] = df['transaction_date'].dt.hour
        df['day_of_month'] = df['transaction_date'].dt.day
        
        frequency_analysis = {
            'daily_patterns': df['day_of_week'].value_counts().to_dict(),
            'hourly_patterns': df['hour'].value_counts().to_dict(),
            'monthly_day_patterns': df['day_of_month'].value_counts().to_dict(),
            'avg_transactions_per_day': len(transactions) / max((df['transaction_date'].max() - df['transaction_date'].min()).days, 1),
            'peak_activity_periods': self._identify_peak_periods(df)
        }
        
        return frequency_analysis
    
    def analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze seasonal spending patterns"""
        
        if df.empty:
            return {}
        
        df['month'] = df['transaction_date'].dt.month
        df['quarter'] = df['transaction_date'].dt.quarter
        
        seasonal_analysis = {
            'monthly_spending': df[df['transaction_type'] == 2].groupby('month')['transaction_amount'].sum().abs().to_dict(),
            'quarterly_patterns': df[df['transaction_type'] == 2].groupby('quarter')['transaction_amount'].sum().abs().to_dict(),
            'seasonal_trends': self._identify_seasonal_trends(df)
        }
        
        return seasonal_analysis
    
    def detect_transaction_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect unusual transactions (outliers)"""
        
        if df.empty:
            return {}
        
        amounts = df['transaction_amount'].abs()
        
        # Using IQR method for outlier detection
        Q1 = amounts.quantile(0.25)
        Q3 = amounts.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(amounts < lower_bound) | (amounts > upper_bound)]
        
        return {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(df) * 100,
            'high_value_outliers': outliers[outliers['transaction_amount'].abs() > upper_bound].to_dict('records'),
            'outlier_threshold': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    def analyze_transaction_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze transaction trends over time"""
        
        if df.empty:
            return {}
        
        # Monthly trends
        df['month'] = df['transaction_date'].dt.to_period('M')
        monthly_spending = df[df['transaction_type'] == 2].groupby('month')['transaction_amount'].sum().abs()
        monthly_income = df[df['transaction_type'] == 1].groupby('month')['transaction_amount'].sum()
        
        trends = {
            'spending_trend': self._calculate_trend(monthly_spending),
            'income_trend': self._calculate_trend(monthly_income),
            'net_cash_flow_trend': self._calculate_trend(monthly_income - monthly_spending),
            'transaction_volume_trend': self._calculate_trend(df.groupby('month').size())
        }
        
        return trends
    
    def detect_recurring_transactions(self, transactions: List[Dict]) -> List[Dict]:
        """Detect recurring transaction patterns"""
        
        # Group by similar amounts and narrations
        amount_groups = defaultdict(list)
        
        for txn in transactions:
            # Round amount to nearest 100 for grouping
            rounded_amount = round(float(txn['transaction_amount']), -2)
            amount_groups[rounded_amount].append(txn)
        
        recurring_patterns = []
        
        for amount, txn_group in amount_groups.items():
            if len(txn_group) >= 3:  # At least 3 occurrences
                # Check if narrations are similar
                narrations = [t.get('transaction_narration', '') for t in txn_group]
                if self._are_narrations_similar(narrations):
                    dates = [datetime.strptime(t['transaction_date'], '%Y-%m-%d') for t in txn_group]
                    date_diffs = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
                    avg_interval = np.mean(date_diffs)
                    
                    recurring_patterns.append({
                        'pattern_type': 'recurring_payment' if amount < 0 else 'recurring_income',
                        'amount': abs(amount),
                        'frequency': len(txn_group),
                        'avg_interval_days': avg_interval,
                        'sample_narration': narrations[0],
                        'estimated_monthly_impact': abs(amount) * (30 / avg_interval) if avg_interval > 0 else 0
                    })
        
        return recurring_patterns
    
    def calculate_transaction_metrics(self, transactions: List[Dict]) -> Dict:
        """Calculate key transaction metrics"""
        
        if not transactions:
            return {}
        
        amounts = [abs(float(t['transaction_amount'])) for t in transactions]
        credit_amounts = [float(t['transaction_amount']) for t in transactions if t.get('transaction_type') == 1]
        debit_amounts = [abs(float(t['transaction_amount'])) for t in transactions if t.get('transaction_type') == 2]
        
        metrics = {
            'total_transaction_value': sum(amounts),
            'avg_transaction_amount': np.mean(amounts),
            'median_transaction_amount': np.median(amounts),
            'std_transaction_amount': np.std(amounts),
            'total_credits': sum(credit_amounts),
            'total_debits': sum(debit_amounts),
            'net_cash_flow': sum(credit_amounts) - sum(debit_amounts),
            'transaction_count_by_type': {
                'credit': len(credit_amounts),
                'debit': len(debit_amounts)
            },
            'largest_transaction': max(amounts) if amounts else 0,
            'smallest_transaction': min(amounts) if amounts else 0
        }
        
        return metrics
    
    # Helper methods
    def _get_transaction_type_name(self, txn_type: int) -> str:
        """Get transaction type name"""
        type_map = {
            1: 'credit',
            2: 'debit',
            3: 'opening',
            4: 'interest',
            5: 'tds',
            6: 'installment',
            7: 'closing',
            8: 'others'
        }
        return type_map.get(txn_type, 'unknown')
    
    def _analyze_spending_velocity(self, debit_transactions: List[Dict]) -> Dict:
        """Analyze spending velocity patterns"""
        
        if not debit_transactions:
            return {}
        
        df = pd.DataFrame(debit_transactions)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['transaction_amount'] = df['transaction_amount'].abs()
        
        daily_spending = df.groupby(df['transaction_date'].dt.date)['transaction_amount'].sum()
        
        return {
            'avg_daily_spending': daily_spending.mean(),
            'max_daily_spending': daily_spending.max(),
            'spending_variance': daily_spending.var(),
            'high_spending_days': len(daily_spending[daily_spending > daily_spending.quantile(0.8)])
        }
    
    def _calculate_income_stability(self, monthly_income: pd.Series) -> float:
        """Calculate income stability score (0-100)"""
        
        if len(monthly_income) < 2:
            return 0
        
        # Calculate coefficient of variation
        cv = monthly_income.std() / monthly_income.mean() if monthly_income.mean() > 0 else 1
        
        # Convert to stability score (lower CV = higher stability)
        stability_score = max(0, 100 - (cv * 100))
        return min(stability_score, 100)
    
    def _analyze_salary_patterns(self, salary_transactions: List[Dict]) -> Dict:
        """Analyze salary payment patterns"""
        
        if not salary_transactions:
            return {}
        
        amounts = [float(t['transaction_amount']) for t in salary_transactions]
        dates = [datetime.strptime(t['transaction_date'], '%Y-%m-%d') for t in salary_transactions]
        
        return {
            'avg_salary': np.mean(amounts),
            'salary_variance': np.var(amounts),
            'payment_frequency': len(salary_transactions),
            'typical_pay_day': self._find_typical_pay_day(dates)
        }
    
    def _find_typical_pay_day(self, dates: List[datetime]) -> int:
        """Find typical pay day of month"""
        days = [d.day for d in dates]
        return max(set(days), key=days.count) if days else 0
    
    def _identify_peak_periods(self, df: pd.DataFrame) -> Dict:
        """Identify peak activity periods"""
        
        hourly_counts = df['hour'].value_counts()
        daily_counts = df['day_of_week'].value_counts()
        
        return {
            'peak_hour': hourly_counts.index[0] if not hourly_counts.empty else None,
            'peak_day': daily_counts.index[0] if not daily_counts.empty else None,
            'peak_hour_activity': int(hourly_counts.iloc[0]) if not hourly_counts.empty else 0,
            'peak_day_activity': int(daily_counts.iloc[0]) if not daily_counts.empty else 0
        }
    
    def _identify_seasonal_trends(self, df: pd.DataFrame) -> Dict:
        """Identify seasonal spending trends"""
        
        monthly_spending = df[df['transaction_type'] == 2].groupby('month')['transaction_amount'].sum().abs()
        
        if len(monthly_spending) < 3:
            return {}
        
        return {
            'highest_spending_month': int(monthly_spending.idxmax()),
            'lowest_spending_month': int(monthly_spending.idxmin()),
            'spending_seasonality': 'high' if monthly_spending.std() > monthly_spending.mean() * 0.3 else 'low'
        }
    
    def _calculate_trend(self, series: pd.Series) -> Dict:
        """Calculate trend for a time series"""
        
        if len(series) < 2:
            return {'direction': 'insufficient_data', 'strength': 0}
        
        # Simple linear trend calculation
        x = np.arange(len(series))
        y = series.values
        
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
        
        direction = 'increasing' if correlation > 0.1 else 'decreasing' if correlation < -0.1 else 'stable'
        strength = abs(correlation)
        
        return {
            'direction': direction,
            'strength': strength,
            'correlation': correlation
        }
    
    def _are_narrations_similar(self, narrations: List[str]) -> bool:
        """Check if transaction narrations are similar"""
        
        if len(narrations) < 2:
            return False
        
        # Simple similarity check - if they share common keywords
        common_words = set()
        for narration in narrations:
            words = set(narration.lower().split())
            if not common_words:
                common_words = words
            else:
                common_words = common_words.intersection(words)
        
        return len(common_words) >= 2  # At least 2 common words
    
    def _create_empty_analysis(self) -> Dict:
        """Create empty analysis structure"""
        return {
            'total_transactions': 0,
            'error': 'No transactions to analyze',
            'analysis_timestamp': datetime.now().isoformat()
        }

# Test function
def test_transaction_analyzer():
    """Test the transaction analyzer"""
    print("Testing Transaction Analyzer...")
    
    # Sample transaction data
    sample_transactions = [
        {
            'transaction_amount': -5000,
            'transaction_narration': 'UPI-SWIGGY-SWIGGY8@YBL',
            'transaction_date': '2025-01-15',
            'transaction_type': 2
        },
        {
            'transaction_amount': 80000,
            'transaction_narration': 'SALARY CREDIT',
            'transaction_date': '2025-01-01',
            'transaction_type': 1
        },
        {
            'transaction_amount': -2000,
            'transaction_narration': 'UPI-CRED-CREDIT CARD PAYMENT',
            'transaction_date': '2025-01-10',
            'transaction_type': 2
        }
    ]
    
    analyzer = TransactionAnalyzer()
    results = analyzer.analyze_transaction_patterns(sample_transactions)
    
    print(f" Analysis Results:")
    print(f"Total Transactions: {results['total_transactions']}")
    print(f"Categories Found: {len(results['transaction_categorization']['by_category'])}")
    print(f"Income Stability: {results['income_patterns']['income_stability_score']:.1f}")
    
    return results

if __name__ == "__main__":
    test_transaction_analyzer()
