cat > tests/sql/bigquery/cleanup_test_data.sql << 'EOF'
-- Cleanup Test Data
-- Use this to reset test environment

-- Remove test calculation inputs
DELETE FROM `fiscal-fox-fin.fiscal_master_dw.goal_calculation_inputs`
WHERE master_uid LIKE 'ff_test_%' OR master_uid = 'ff_user_8a838f3528819407';

-- Remove test goals
DELETE FROM `fiscal-fox-fin.fiscal_master_dw.user_goals_enhanced`
WHERE master_uid LIKE 'ff_test_%' OR master_uid = 'ff_user_8a838f3528819407';

-- Verify cleanup
SELECT 
  'goal_calculation_inputs' as table_name,
  COUNT(*) as remaining_test_records
FROM `fiscal-fox-fin.fiscal_master_dw.goal_calculation_inputs`
WHERE master_uid LIKE 'ff_%'

UNION ALL

SELECT 
  'user_goals_enhanced' as table_name,
  COUNT(*) as remaining_test_records
FROM `fiscal-fox-fin.fiscal_master_dw.user_goals_enhanced`
WHERE master_uid LIKE 'ff_%';
EOF
