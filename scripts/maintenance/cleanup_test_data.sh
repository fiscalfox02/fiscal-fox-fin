cat > scripts/maintenance/cleanup_test_data.sh << 'EOF'
#!/bin/bash
set -e

echo "🧹 Cleaning up Goal Engine test data..."

# Execute cleanup SQL
bq query --use_legacy_sql=false < tests/sql/bigquery/cleanup_test_data.sql

echo "✅ Test data cleanup completed"
EOF

chmod +x scripts/maintenance/cleanup_test_data.sh
