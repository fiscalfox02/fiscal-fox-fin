import math
from datetime import datetime, timedelta
from google.cloud import bigquery
import json
import uuid

class GoalCalculator:
    def __init__(self, project_id="fiscal-fox-fin"):
        self.project_id = project_id
        self.client = bigquery.Client()
        self.dataset_id = "fiscal_master_dw"
    
    # Your existing mathematical functions (keep intact)
    def amortization_payment(self, principal, annual_rate, months):
        """Your existing amortization logic"""
        r = annual_rate / 12 / 100
        if r == 0:
            return principal / months
        return (principal * r) / (1 - (1 + r) ** -months)
    
    def set_timeline_goal(self, debts, target_date, monthly_surplus):
        """Your existing timeline goal logic"""
        today = datetime.today()
        total_months = (target_date.year - today.year) * 12 + (target_date.month - today.month)
        payments = []
        for loan in debts:
            P = loan['balance']
            rate = loan['rateOfInterest']
            n = total_months
            pay = self.amortization_payment(P, rate, n)
            payments.append(pay)
        total_required = sum(payments)
        feasible = total_required <= monthly_surplus
        return {
            'goal_type': 'Timeline Goal',
            'months_to_target': total_months,
            'required_monthly_payment': total_required,
            'feasible_with_current_surplus': feasible,
            'surplus_available': monthly_surplus - total_required
        }
    
    def set_investment_goal(self, target_amount, horizon_years, return_rate=0.12, existing_sip=0):
        """Your existing investment goal logic"""
        months = int(horizon_years * 12)
        r = return_rate / 12
        factor = ((1+r) ** months - 1) / r
        sip = target_amount / factor
        return {
            'goal_type': 'Investment Goal',
            'horizon_months': months,
            'required_monthly_sip': sip,
            'total_sip_post_allocation': existing_sip + sip
        }
    
    def set_emergency_fund_goal(self, avg_monthly_expense, income_type='salaried', current_emis=0, expense_volatility=0):
        """Your existing emergency fund logic"""
        base = avg_monthly_expense
        multiplier_map = {'salaried': 6, 'gig': 9, 'contract': 8}
        mult = multiplier_map.get(income_type, 6)
        if expense_volatility > 0.2:
            mult += 1
        target = base * mult + current_emis
        return {
            'goal_type': 'Emergency Fund Goal',
            'months_of_cover': mult,
            'target_amount': target
        }
    
    # BigQuery integration functions
    def get_user_inputs(self, master_uid):
        """Fetch user's calculation inputs from BigQuery"""
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.goal_calculation_inputs`
        WHERE master_uid = '{master_uid}'
        ORDER BY calculation_timestamp DESC
        LIMIT 1
        """
        results = self.client.query(query).to_dataframe()
        return results.iloc[0].to_dict() if not results.empty else None
    
    def save_goal_to_bq(self, goal_data, master_uid):
        """Save calculated goal to BigQuery"""
        goal_id = str(uuid.uuid4())
        
        # Prepare row for BigQuery insertion
        row = {
            'master_uid': master_uid,
            'goal_id': goal_id,
            'goal_type': goal_data.get('goal_type', '').lower().replace(' ', '_'),
            'goal_name': goal_data.get('goal_name'),
            'goal_description': goal_data.get('goal_description'),
            'target_date': goal_data.get('target_date'),
            'months_to_target': goal_data.get('months_to_target'),
            'required_monthly_payment': goal_data.get('required_monthly_payment'),
            'target_amount': goal_data.get('target_amount'),
            'horizon_years': goal_data.get('horizon_years'),
            'expected_return_rate': goal_data.get('expected_return_rate', 0.12),
            'required_monthly_sip': goal_data.get('required_monthly_sip'),
            'feasible_with_current_surplus': goal_data.get('feasible_with_current_surplus'),
            'surplus_remaining_after_goal': goal_data.get('surplus_available'),
            'feasibility_confidence_score': goal_data.get('feasibility_confidence_score', 0.5),
            'status': 'active',
            'created_at': datetime.now()
        }
        
        # Insert into BigQuery
        table_id = f"{self.project_id}.{self.dataset_id}.user_goals_enhanced"
        table = self.client.get_table(table_id)
        errors = self.client.insert_rows_json(table, [row])
        
        if not errors:
            return goal_id
        else:
            raise Exception(f"BigQuery insertion failed: {errors}")
