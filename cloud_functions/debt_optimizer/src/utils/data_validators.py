from typing import List, Dict, Any
import re

class DebtDataValidator:
    def __init__(self):
        self.required_debt_fields = [
            'debt_name', 'debt_type', 'total_outstanding', 
            'current_balance', 'interest_rate', 'min_payment'
        ]
    
    def validate_debt_portfolio(self, debt_portfolio: List[Dict]) -> bool:
        """Validate debt portfolio data"""
        if not isinstance(debt_portfolio, list) or not debt_portfolio:
            print("Debt portfolio must be a non-empty list")
            return False
        
        for i, debt in enumerate(debt_portfolio):
            if not self.validate_single_debt(debt, i):
                return False
        
        return True
    
    def validate_single_debt(self, debt: Dict, index: int = 0) -> bool:
        """Validate a single debt entry"""
        # Check required fields
        for field in self.required_debt_fields:
            if field not in debt:
                print(f"Missing required field '{field}' in debt {index}")
                return False
        
        # Validate data types and ranges
        try:
            # Numeric validations
            if debt['total_outstanding'] < 0:
                print(f"Total outstanding cannot be negative in debt {index}")
                return False
            
            if debt['interest_rate'] < 0 or debt['interest_rate'] > 1:
                print(f"Interest rate must be between 0 and 1 in debt {index}")
                return False
            
            if debt['min_payment'] < 0:
                print(f"Minimum payment cannot be negative in debt {index}")
                return False
            
            # String validations
            if not isinstance(debt['debt_name'], str) or not debt['debt_name'].strip():
                print(f"Debt name must be a non-empty string in debt {index}")
                return False
            
            return True
            
        except (TypeError, ValueError) as e:
            print(f"Data type error in debt {index}: {e}")
            return False
    
    def sanitize_debt_data(self, debt_portfolio: List[Dict]) -> List[Dict]:
        """Sanitize and clean debt data"""
        sanitized = []
        
        for debt in debt_portfolio:
            sanitized_debt = {
                'debt_name': str(debt['debt_name']).strip(),
                'debt_type': str(debt.get('debt_type', 'Unknown')).strip(),
                'total_outstanding': float(debt['total_outstanding']),
                'current_balance': float(debt.get('current_balance', debt['total_outstanding'])),
                'past_due': float(debt.get('past_due', 0)),
                'interest_rate': float(debt['interest_rate']),
                'min_payment': float(debt['min_payment']),
                'credit_limit': float(debt.get('credit_limit', 0)),
                'portfolio_type': str(debt.get('portfolio_type', 'R')).upper(),
                'payment_rating': int(debt.get('payment_rating', 1)),
                'account_status': str(debt.get('account_status', 'Active')),
                'payment_history': str(debt.get('payment_history', 'Regular'))
            }
            sanitized.append(sanitized_debt)
        
        return sanitized
"""check"""
