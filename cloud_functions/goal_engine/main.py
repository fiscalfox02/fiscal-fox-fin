import functions_framework
from goal_calculator import GoalCalculator
from datetime import datetime
import json

@functions_framework.http
def create_goal_endpoint(request):
    """Main endpoint for creating goals using your existing logic"""
    # Enable CORS if needed
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    headers = {'Access-Control-Allow-Origin': '*'}
    
    try:
        # Parse request
        request_json = request.get_json(silent=True)
        if not request_json:
            return ({'error': 'No JSON data provided'}, 400, headers)
        
        # Initialize calculator with your logic
        calculator = GoalCalculator()
        
        # Extract parameters
        master_uid = request_json.get('master_uid', 'ff_user_8a838f3528819407')
        goal_type = request_json.get('goal_type')
        goal_name = request_json.get('goal_name')
        
        # Get user's financial inputs from BigQuery
        user_inputs = calculator.get_user_inputs(master_uid)
        if not user_inputs:
            return ({'error': 'No user inputs found in BigQuery'}, 404, headers)
        
        # Route to appropriate goal calculation
        if goal_type == 'timeline':
            result = handle_timeline_goal(request_json, calculator, user_inputs, master_uid, goal_name)
        elif goal_type == 'investment':
            result = handle_investment_goal(request_json, calculator, user_inputs, master_uid, goal_name)
        elif goal_type == 'emergency':
            result = handle_emergency_goal(request_json, calculator, user_inputs, master_uid, goal_name)
        else:
            return ({'error': f'Unsupported goal_type: {goal_type}'}, 400, headers)
        
        return (result, 200, headers)
        
    except Exception as e:
        return ({'error': str(e)}, 500, headers)

def handle_timeline_goal(request_json, calculator, user_inputs, master_uid, goal_name):
    """Handle timeline goal using your existing amortization logic"""
    # Extract timeline-specific parameters
    target_date_str = request_json.get('target_date')
    if not target_date_str:
        raise ValueError("target_date is required for timeline goals")
    
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
    
    # Use debt data from BigQuery or request
    debts = request_json.get('debts', [])
    if not debts and user_inputs.get('debt_accounts_json'):
        # Parse debts from BigQuery
        debt_accounts = json.loads(user_inputs['debt_accounts_json'])
        debts = [
            {
                'balance': debt['balance'],
                'rateOfInterest': debt['interest_rate']
            }
            for debt in debt_accounts
        ]
    
    monthly_surplus = user_inputs['available_monthly_surplus']
    
    # Apply your existing timeline goal logic
    goal_result = calculator.set_timeline_goal(debts, target_date, monthly_surplus)
    
    # Enhance with metadata
    goal_data = {
        **goal_result,
        'goal_name': goal_name,
        'goal_description': f"Clear all debts by {target_date.strftime('%B %Y')}",
        'target_date': target_date.date(),
        'feasibility_confidence_score': calculate_confidence_score(goal_result, monthly_surplus)
    }
    
    # Save to BigQuery
    goal_id = calculator.save_goal_to_bq(goal_data, master_uid)
    
    return {
        'success': True,
        'goal_id': goal_id,
        'calculation_result': goal_result,
        'user_inputs_used': {
            'monthly_surplus': monthly_surplus,
            'debts_count': len(debts),
            'total_debt': sum(d['balance'] for d in debts)
        }
    }

def handle_investment_goal(request_json, calculator, user_inputs, master_uid, goal_name):
    """Handle investment goal using your existing SIP logic"""
    target_amount = request_json.get('target_amount')
    horizon_years = request_json.get('horizon_years')
    
    if not target_amount or not horizon_years:
        raise ValueError("target_amount and horizon_years required for investment goals")
    
    existing_sip = user_inputs.get('current_sip_investments', 0)
    
    # Apply your existing investment goal logic
    goal_result = calculator.set_investment_goal(target_amount, horizon_years, 0.12, existing_sip)
    
    # Check feasibility against surplus
    monthly_surplus = user_inputs['available_monthly_surplus']
    feasible = goal_result['required_monthly_sip'] <= monthly_surplus
    
    goal_data = {
        **goal_result,
        'goal_name': goal_name,
        'goal_description': f"Build ₹{target_amount:,.0f} corpus in {horizon_years} years",
        'target_amount': target_amount,
        'horizon_years': horizon_years,
        'expected_return_rate': 0.12,
        'feasible_with_current_surplus': feasible,
        'surplus_available': monthly_surplus - goal_result['required_monthly_sip']
    }
    
    # Save to BigQuery
    goal_id = calculator.save_goal_to_bq(goal_data, master_uid)
    
    return {
        'success': True,
        'goal_id': goal_id,
        'calculation_result': goal_result,
        'feasibility_check': {
            'required_sip': goal_result['required_monthly_sip'],
            'available_surplus': monthly_surplus,
            'feasible': feasible
        }
    }

def handle_emergency_goal(request_json, calculator, user_inputs, master_uid, goal_name):
    """Handle emergency fund using your existing logic"""
    avg_monthly_expense = user_inputs.get('monthly_fixed_expenses', 0) + user_inputs.get('monthly_variable_expenses', 0)
    income_type = request_json.get('income_type', 'salaried')
    current_emis = user_inputs.get('current_emi_obligations', 0)
    
    # Apply your existing emergency fund logic
    goal_result = calculator.set_emergency_fund_goal(avg_monthly_expense, income_type, current_emis)
    
    goal_data = {
        **goal_result,
        'goal_name': goal_name,
        'goal_description': f"{goal_result['months_of_cover']} months emergency fund",
        'emergency_target_calculated': goal_result['target_amount']
    }
    
    # Save to BigQuery
    goal_id = calculator.save_goal_to_bq(goal_data, master_uid)
    
    return {
        'success': True,
        'goal_id': goal_id,
        'calculation_result': goal_result
    }

def calculate_confidence_score(goal_result, monthly_surplus):
    """Calculate confidence score based on surplus availability"""
    if not goal_result.get('feasible_with_current_surplus'):
        return 0.2
    
    surplus_ratio = goal_result.get('surplus_available', 0) / monthly_surplus
    if surplus_ratio > 0.5:
        return 0.9
    elif surplus_ratio > 0.2:
        return 0.7
    else:
        return 0.5

@functions_framework.http
def get_goals_endpoint(request):
    """Endpoint to fetch user's goals from BigQuery"""
    headers = {'Access-Control-Allow-Origin': '*'}
    
    try:
        master_uid = request.args.get('master_uid', 'ff_user_8a838f3528819407')
        
        calculator = GoalCalculator()
        
        query = f"""
        SELECT *
        FROM `{calculator.project_id}.{calculator.dataset_id}.user_goals_enhanced`
        WHERE master_uid = '{master_uid}' AND status = 'active'
        ORDER BY created_at DESC
        """
        
        results = calculator.client.query(query).to_dataframe()
        goals = results.to_dict('records')
        
        # Convert datetime objects to strings for JSON serialization
        for goal in goals:
            for key, value in goal.items():
                if hasattr(value, 'strftime'):
                    goal[key] = value.strftime('%Y-%m-%d') if 'date' in key.lower() else value.isoformat()
        
        return ({
            'success': True,
            'goals': goals,
            'total_count': len(goals)
        }, 200, headers)
        
    except Exception as e:
        return ({'error': str(e)}, 500, headers)
