#  Fiscal Fox Debt Optimizer

A comprehensive debt analysis and optimization system for intelligent financial planning.

##  Features

- **Advanced Debt Analysis**: Priority scoring using avalanche and snowball methods
- **Financial Health Assessment**: Complete financial strength scoring
- **Smart Recommendations**: AI-powered debt payoff strategies
- **BigQuery Integration**: Scalable data storage and analytics
- **API-First Design**: Easy integration with applications

##  Quick Start

### Prerequisites
- Python 3.9+
- Google Cloud Platform account
- gcloud CLI installed

### Setup

1. **Clone and navigate to the debt optimizer:**
cd fiscal-fox-fin/cloud_functions/debt_optimizer
   
2. **Install dependencies:**
pip install -r requirements-dev.txt

3. **Set up GCP resources:**
chmod +x scripts/*.sh
./scripts/setup_gcp_resources.sh


4. **Create BigQuery tables:**
./scripts/create_bigquery_tables.sh


5. **Run tests:**
python -m pytest tests/ -v

6. **Test locally:**
python -c "from src.enhanced_debt_optimizer import test_debt_optimizer; test_debt_optimizer()"


## Deployment

### Via GitHub Actions (Recommended)
1. Set up GitHub Secrets:
- `GCP_SA_KEY`: Service account JSON key
- `GCP_PROJECT_ID`: fiscal-fox-fin

2. Push to main branch:
git add .
git commit -m "deploy: debt optimizer setup"
git push origin main

### Manual Deployment
./scripts/deploy_functions.sh

##  API Usage
curl -X POST "https://us-central1-fiscal-fox-fin.cloudfunctions.net/debt-analyzer-api"
-H "Content-Type: application/json"
-d '{"user_id": "test_user_123"}'

##  Testing
Run all tests
python -m pytest tests/ -v

Run with coverage
python -m pytest tests/ --cov=src/

##  Documentation

- [API Documentation](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Architecture Overview](docs/ARCHITECTURE.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License.





  


