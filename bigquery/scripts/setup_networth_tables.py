#!/usr/bin/env python3
"""
Setup Net Worth Analysis Tables for Fiscal Fox
"""

import os
import subprocess
from google.cloud import bigquery

def setup_networth_tables():
    """Create all net worth analysis tables"""
    
    print("Setting up Net Worth Analysis Tables")
    print("=" * 50)
    
    client = bigquery.Client(project="fiscal-fox-in")
    
    # List of SQL files to execute
    sql_files = [
        "core_tables/networth_results.sql",
        "core_tables/networth_webhook.sql"
    ]
    
    for sql_file in sql_files:
        if os.path.exists(sql_file):
            print(f"Creating table from {sql_file}...")
            
            with open(sql_file, 'r') as f:
                sql_content = f.read()
            
            try:
                query_job = client.query(sql_content)
                query_job.result()  # Wait for completion
                print(f"✅ {sql_file} executed successfully")
            except Exception as e:
                print(f" Failed to execute {sql_file}: {e}")
        else:
            print(f" File not found: {sql_file}")
    
    # Verify tables were created
    verify_networth_tables(client)

def verify_networth_tables(client):
    """Verify net worth tables were created"""
    
    query = """
    SELECT table_name, creation_time
    FROM `fiscal-fox-in.fiscal_master_dw.INFORMATION_SCHEMA.TABLES`
    WHERE table_name IN ('networth_results', 'networth_webhook')
    ORDER BY table_name
    """
    
    print("\n Verifying created net worth tables:")
    results = client.query(query).result()
    
    for row in results:
        print(f"  {row.table_name} (created: {row.creation_time})")

if __name__ == "__main__":
    setup_networth_tables()
