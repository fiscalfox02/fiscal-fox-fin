import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import os
import yaml

from .utils.bigquery_client import BigQueryManager
from .utils.data_validators import DebtDataValidator

class EnhancedDebtOptimizer:
    def __init__(self, config_path: str = None, use_bigquery: bool = True):
        """Initialize the Enhanced Debt Optimizer"""
        self.config = self._load_config(config_path)
        self.use_bigquery = use_bigquery
        self.validator = DebtDataValidator()
        
        # Initialize BigQuery manager
        if self.use_bigquery:
            self.bq_manager = BigQueryManager(
                project_id=self.config.get('gcp', {}).get('project_id'),
                dataset_id=self.config.get('bigquery', {}).get('dataset_id', 'debt_analytics')
            )
        
        # Core data structures
        self.financial_data = {}
        self.debt_portfolio = []
        self.asset_portfolio = {}
        self.financial_ratios = {}
        self.optimization_results = {}
        self.user_id = None
    
    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration from YAML file"""
        if config_path is None:
            env = os.getenv('ENV', 'development')
            config_path = f"config/{env}.yaml"
        
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'bigquery': {'dataset_id': 'debt_analytics'},
            'optimization': {
                'min_emergency_fund_months': 3,
                'high_interest_threshold': 0.12
            }
        }
    
    def set_user_context(self, user_id: str):
        """Set user context"""
        self.user_id = user_id
        return self.user_id
    
    def load_sample_data(self):
        """Load sample data for testing"""
        self.debt_portfolio = [
            {
                'debt_name': 'Credit Card ABC',
                'debt_type': 'Credit Card',
                'total_outstanding': 75000,
                'current_balance': 75000,
                'past_due': 0,
                'interest_rate': 0.18,
                'min_payment': 3750,
                'credit_limit': 150000,
                'portfolio_type': 'R',
                'payment_rating': 1,
                'account_status': 'Active',
                'payment_history': 'Regular'
            },
            {
                'debt_name': 'Personal Loan XYZ',
                'debt_type': 'Loan',
                'total_outstanding': 200000,
                'current_balance': 200000,
                'past_due': 0,
                'interest_rate': 0.14,
                'min_payment': 8000,
                'credit_limit': 0,
                'portfolio_type': 'I',
                'payment_rating': 2,
                'account_status': 'Active',
                'payment_history': 'Regular'
            }
        ]
        
        self.financial_ratios = {
            'total_assets': 800000,
            'total_debt': 275000,
            'net_worth': 525000,
            'financial_strength_score': 75,
            'estimated_monthly_income': 85000,
            'debt_to_income_ratio': 0.28
        }
        
        self.asset_portfolio = {
            'savings_account': 25000,
            'mutual_funds': 125000,
            'epf': 200000
        }
    
    def calculate_advanced_priority_score(self, debt):
        """Calculate priority score for debt"""
        try:
            # Base components
            avalanche_component = debt['interest_rate'] * 0.4
            balance_ratio = debt['total_outstanding'] / max(sum(d['total_outstanding'] for d in self.debt_portfolio), 1)
            snowball_component = (1 - balance_ratio) * 0.2
            
            # Risk factors
            past_due_penalty = 0.3 if debt['past_due'] > 0 else 0
            high_utilization_penalty = 0.1 if (debt['credit_limit'] > 0 and debt['current_balance'] / debt['credit_limit'] > 0.8) else 0
            
            # Payment rating adjustment
            payment_rating_bonus = (4 - debt['payment_rating']) * 0.05
            
            priority_score = min(avalanche_component + snowball_component + past_due_penalty + 
                               high_utilization_penalty + payment_rating_bonus, 1.0)
            
            components = {
                'avalanche_component': avalanche_component,
                'snowball_component': snowball_component,
                'past_due_penalty': past_due_penalty,
                'high_utilization_penalty': high_utilization_penalty,
                'payment_rating_bonus': payment_rating_bonus
            }
            
            return priority_score, components
            
        except Exception as e:
            print(f"Error calculating priority score: {e}")
            return 0.5, {}
    
    def optimize_debt_with_surplus_analysis(self):
        """Perform comprehensive debt optimization"""
        if not self.debt_portfolio:
            print("No debt portfolio data available")
            return {}
        
        try:
            debt_priorities = []
            total_monthly_payments = 0
            
            for debt in self.debt_portfolio:
                priority_score, components = self.calculate_advanced_priority_score(debt)
                
                base_payment = debt['min_payment']
                total_monthly_payments += base_payment
                
                # Calculate payoff timeline
                if debt['interest_rate'] > 0:
                    monthly_interest_rate = debt['interest_rate'] / 12
                    months_to_payoff = np.log(1 + (debt['total_outstanding'] * monthly_interest_rate) / base_payment) / np.log(1 + monthly_interest_rate)
                    total_interest_paid = (base_payment * months_to_payoff) - debt['total_outstanding']
                else:
                    months_to_payoff = debt['total_outstanding'] / base_payment
                    total_interest_paid = 0
                
                debt_info = {
                    'debt': debt,
                    'priority_score': priority_score,
                    'priority_components': components,
                    'monthly_payment': base_payment,
                    'months_to_payoff': max(months_to_payoff, 1),
                    'total_interest_paid': max(total_interest_paid, 0),
                    'extra_payment': 0
                }
                debt_priorities.append(debt_info)
            
            # Sort by priority score
            debt_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Calculate available surplus
            monthly_income = self.financial_ratios.get('estimated_monthly_income', 0)
            available_surplus = max(monthly_income - total_monthly_payments - 25000, 0)
            
            # Allocate surplus to highest priority debts
            remaining_surplus = available_surplus
            for debt_info in debt_priorities:
                if remaining_surplus > 1000:
                    extra_payment = min(remaining_surplus * 0.6, debt_info['debt']['total_outstanding'] * 0.1)
                    debt_info['extra_payment'] = extra_payment
                    debt_info['monthly_payment'] += extra_payment
                    remaining_surplus -= extra_payment
            
            # Generate recommendations
            recommendations = self._generate_recommendations(debt_priorities, available_surplus)
            
            # Compile results
            self.optimization_results = {
                'financial_summary': {
                    'net_worth': self.financial_ratios.get('net_worth', 0),
                    'total_debt': sum(d['total_outstanding'] for d in self.debt_portfolio),
                    'financial_strength_score': self.financial_ratios.get('financial_strength_score', 0),
                    'available_surplus': available_surplus,
                    'total_monthly_payments': total_monthly_payments
                },
                'debt_priorities': debt_priorities,
                'recommendations': recommendations,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            return self.optimization_results
            
        except Exception as e:
            print(f"Error in debt optimization: {e}")
            return {}
    
    def _generate_recommendations(self, debt_priorities, available_surplus):
        """Generate recommendations"""
        recommendations = {
            'executive_summary': [],
            'priority_actions': [],
            'emergency_actions': []
        }
        
        strength_score = self.financial_ratios.get('financial_strength_score', 0)
        if strength_score > 70:
            recommendations['executive_summary'].append(
                f"Strong Financial Position (Score: {strength_score}/100): You can afford aggressive debt payoff strategies"
            )
        else:
            recommendations['executive_summary'].append(
                f"Financial Position (Score: {strength_score}/100): Focus on high-priority debts while maintaining stability"
            )
        
        if debt_priorities:
            top_debt = debt_priorities[0]
            recommendations['priority_actions'].append(
                f"Focus on eliminating {top_debt['debt']['debt_name']} first"
            )
        
        # Emergency actions for past due debts
        emergency_debts = [d for d in debt_priorities if d['debt']['past_due'] > 0]
        for debt_info in emergency_debts:
            recommendations['emergency_actions'].append(
                f"URGENT: Pay ₹{debt_info['debt']['past_due']:,.0f} past due amount for {debt_info['debt']['debt_name']}"
            )
        
        return recommendations
    
    def create_webhook_response(self):
        """Create webhook-ready response"""
        if not self.optimization_results:
            return {"error": "No optimization results available"}
        
        try:
            summary = self.optimization_results['financial_summary']
            debt_priorities = self.optimization_results['debt_priorities']
            recommendations = self.optimization_results['recommendations']
            
            webhook_response = {
                'user_id': self.user_id,
                'timestamp': datetime.now().isoformat(),
                'status': 'success',
                'financial_health': {
                    'net_worth': summary['net_worth'],
                    'total_debt': summary['total_debt'],
                    'financial_strength_score': summary['financial_strength_score'],
                    'strength_level': self._get_strength_level(summary['financial_strength_score']),
                    'available_surplus': summary['available_surplus']
                },
                'debt_summary': {
                    'total_debts': len(debt_priorities),
                    'highest_priority_debt': debt_priorities[0]['debt']['debt_name'] if debt_priorities else None,
                    'total_monthly_payments': summary['total_monthly_payments']
                },
                'top_recommendations': recommendations.get('executive_summary', [])[:3],
                'priority_debts': [
                    {
                        'debt_name': debt_info['debt']['debt_name'],
                        'amount': debt_info['debt']['total_outstanding'],
                        'interest_rate': debt_info['debt']['interest_rate'],
                        'monthly_payment': debt_info['monthly_payment'],
                        'payoff_months': debt_info.get('months_to_payoff', 0),
                        'priority_rank': idx + 1,
                        'urgency_level': self._get_urgency_level(debt_info['priority_score'])
                    }
                    for idx, debt_info in enumerate(debt_priorities[:5])
                ],
                'action_items': self._extract_action_items(),
                'ui_alerts': self._generate_ui_alerts()
            }
            
            return webhook_response
            
        except Exception as e:
            print(f"Error creating webhook response: {e}")
            return {"error": str(e)}
    
    def _get_strength_level(self, score):
        """Convert score to level"""
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _get_urgency_level(self, priority_score):
        """Convert priority score to urgency"""
        if priority_score >= 0.8:
            return "Critical"
        elif priority_score >= 0.6:
            return "High"
        elif priority_score >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def _extract_action_items(self):
        """Extract actionable items"""
        actions = []
        if not self.optimization_results:
            return actions
        
        debt_priorities = self.optimization_results['debt_priorities']
        recommendations = self.optimization_results['recommendations']
        
        # Emergency actions
        for action in recommendations.get('emergency_actions', []):
            actions.append({
                'type': 'emergency',
                'title': 'Urgent Payment Required',
                'description': action,
                'priority': 'critical'
            })
        
        # High priority debt actions
        if debt_priorities:
            top_debt = debt_priorities[0]
            if top_debt['extra_payment'] > 0:
                actions.append({
                    'type': 'payment',
                    'title': f'Increase {top_debt["debt"]["debt_name"]} Payment',
                    'description': f'Add ₹{top_debt["extra_payment"]:,.0f}/month',
                    'priority': 'high'
                })
        
        return actions[:5]
    
    def _generate_ui_alerts(self):
        """Generate UI alerts"""
        alerts = []
        if not self.optimization_results:
            return alerts
        
        debt_priorities = self.optimization_results['debt_priorities']
        
        # Past due alert
        past_due_debts = [d for d in debt_priorities if d['debt']['past_due'] > 0]
        if past_due_debts:
            total_past_due = sum(d['debt']['past_due'] for d in past_due_debts)
            alerts.append({
                'type': 'critical',
                'title': 'Overdue Payments Detected',
                'message': f'You have ₹{total_past_due:,.0f} in overdue payments.',
                'action': 'make_payment_now'
            })
        
        return alerts

# Test function
def test_debt_optimizer():
    """Test the debt optimizer"""
    print(" Testing Enhanced Debt Optimizer...")
    
    optimizer = EnhancedDebtOptimizer(use_bigquery=False)
    optimizer.set_user_context("test_user_12345")
    optimizer.load_sample_data()
    
    results = optimizer.optimize_debt_with_surplus_analysis()
    
    if results:
        print(" Optimization completed successfully!")
        webhook_response = optimizer.create_webhook_response()
        return webhook_response
    else:
        print(" Optimization failed")
        return None

if __name__ == "__main__":
    test_debt_optimizer()
