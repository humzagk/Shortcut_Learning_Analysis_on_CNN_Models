# parse_results.py

import re
import pandas as pd
import sys

# Path to your log file
log_file = sys.argv[1]

results = []
current_exp = {}

with open(log_file, 'r') as f:
    for line in f:
        # Detect Experiment Start
        # Format: "Exp: Train on X -> Test on Y | Model: Z"
        match_exp = re.search(r"Exp: Train on (.*?) -> Test on (.*?) \| Model: (.*)", line)
        if match_exp:
            current_exp = {
                'Model': match_exp.group(3).strip(),
                'Train': match_exp.group(1).strip(),
                'Test': match_exp.group(2).strip()
            }

        # Detect Final Accuracy
        # Format: "Best Accuracy: 0.9958"
        match_acc = re.search(r"Best Accuracy: ([\d\.]+)", line)
        if match_acc and current_exp:
            current_exp['Accuracy'] = float(match_acc.group(1))
            results.append(current_exp)
            current_exp = {}

# Create DataFrame
df = pd.DataFrame(results)
print(df.to_string(index=False))

# Save to CSV
df.to_csv("final_results_summary.csv", index=False)
print("\nSaved to final_results_summary.csv")