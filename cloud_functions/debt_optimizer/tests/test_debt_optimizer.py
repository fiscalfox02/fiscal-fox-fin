import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_debt_optimizer import EnhancedDebtOptimizer

class TestDebtOptimizer:
    @pytest.fixture
    def optimizer(self):
        return EnhancedDebtOptimizer(use_bigquery=False)
    
    def test_initialization(self, optimizer):
        assert optimizer is not None
        assert not optimizer.use_bigquery
    
    def test_sample_data_loading(self, optimizer):
        optimizer.load_sample_data()
        assert len(optimizer.debt_portfolio) > 0
        assert 'financial_strength_score' in optimizer.financial_ratios
    
    def test_debt_optimization(self, optimizer):
        optimizer.set_user_context("test_user")
        optimizer.load_sample_data()
        results = optimizer.optimize_debt_with_surplus_analysis()
        
        assert results is not None
        assert 'financial_summary' in results
        assert 'debt_priorities' in results
    
    def test_webhook_response_creation(self, optimizer):
        optimizer.set_user_context("test_user")
        optimizer.load_sample_data()
        optimizer.optimize_debt_with_surplus_analysis()
        
        response = optimizer.create_webhook_response()
        
        assert 'user_id' in response
        assert 'financial_health' in response
        assert 'debt_summary' in response
        assert response['status'] == 'success'
