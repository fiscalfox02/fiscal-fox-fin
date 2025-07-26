import functions_framework
from flask import jsonify
import json
import os
import sys

# Add the parent directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from enhanced_debt_optimizer import EnhancedDebtOptimizer

@functions_framework.http
def debt_analyzer_api(request):
    """HTTP Cloud Function for debt analysis"""
    
    # Handle CORS
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
        request_json = request.get_json(silent=True)
        if not request_json or 'user_id' not in request_json:
            return jsonify({'error': 'user_id required'}), 400, headers
        
        user_id = request_json['user_id']
        
        # Initialize debt optimizer
        optimizer = EnhancedDebtOptimizer(use_bigquery=True)
        optimizer.set_user_context(user_id)
        
        # Load sample data for testing
        optimizer.load_sample_data()
        
        # Run analysis
        results = optimizer.optimize_debt_with_surplus_analysis()
        
        if results:
            # Save to BigQuery if enabled
            if optimizer.use_bigquery:
                optimizer.bq_manager.save_debt_portfolio(user_id, optimizer.debt_portfolio)
            
            # Create webhook response
            webhook_response = optimizer.create_webhook_response()
            return jsonify(webhook_response), 200, headers
        else:
            return jsonify({'error': 'Analysis failed'}), 500, headers
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500, headers

@functions_framework.http
def health_check(request):
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': str(datetime.now()),
        'service': 'debt-optimizer'
    }), 200
