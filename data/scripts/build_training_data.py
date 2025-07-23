import json
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def parse_units(val):
    if not val:
        return 0
    units = float(val.get("units", 0))
    nanos = int(val.get("nanos", 0)) / 1e9
    return units + nanos

with open(os.path.join(DATA_DIR, "fetch_net_worth.json")) as f:
    net_worth_data = json.load(f)

with open(os.path.join(DATA_DIR, "fetch_credit_report.json")) as f:
    credit_data = json.load(f)



# Assets
asset_list = net_worth_data["netWorthResponse"].get("assetValues", [])
assets_total = sum(parse_units(asset["value"]) for asset in asset_list)

# Liabilities
liability_list = net_worth_data["netWorthResponse"].get("liabilityValues", [])
liabilities_total = sum(parse_units(liability["value"]) for liability in liability_list)

# Credit report metrics
accounts = credit_data["creditReports"][0]["creditReportData"]["creditAccount"]["creditAccountDetails"]
score = float(credit_data["creditReports"][0]["creditReportData"]["score"]["bureauScore"])

interest_rates = []
current_balances = []
amounts_due = []
ratings = []
ratios = []

for acc in accounts:
    try:
        curr = float(acc.get("currentBalance", "0"))
        orig = float(acc.get("highestCreditOrOriginalLoanAmount", "1"))
        due = float(acc.get("amountPastDue", "0"))
        interest = float(acc.get("rateOfInterest", "0"))
        rating = float(acc.get("paymentRating", "0"))

        current_balances.append(curr)
        amounts_due.append(due)
        interest_rates.append(interest)
        ratings.append(rating)
        ratios.append(curr / orig if orig else 0)
    except Exception as e:
        print("Skipping bad account:", e)


income = 70000   
expenses = 45000
monthly_saving = income - expenses
avg_return = 8.5
loan_emi = 20000

projected_assets = assets_total
projected_liabilities = liabilities_total
net_worth_list = []

for month in range(1, 25):
    projected_assets += monthly_saving
    projected_assets *= (1 + 0.007)
    projected_liabilities -= loan_emi
    projected_liabilities = max(0, projected_liabilities)
    net_worth = projected_assets - projected_liabilities
    net_worth_list.append(net_worth)

rule_based_24mo = net_worth_list[-1]

row = {
    "savings_rate": income / expenses,
    "debt_to_asset_ratio": liabilities_total / assets_total if assets_total else 0,
    "avg_return": avg_return,
    "credit_score": score,
    "total_current_balance": np.sum(current_balances),
    "total_amount_past_due": np.sum(amounts_due),
    "avg_interest_rate": np.mean(interest_rates),
    "avg_payment_rating": np.mean(ratings),
    "avg_loan_to_original_ratio": np.mean(ratios),
    "projected_net_worth": rule_based_24mo
}

df = pd.DataFrame([row])
output_path = os.path.join(DATA_DIR, "train_data.csv")
df.to_csv(output_path, index=False)
print(f"✅ Saved dataset to: {output_path}")
