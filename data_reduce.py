import pandas as pd

# Load your data
df = pd.read_csv("D:/MLOps-Student Score predictor/data/students.csv")  # adjust path as needed

print(f"Original rows: {len(df)}")

# Randomly sample 20,000 rows (recommended - avoids bias from row order)
df_reduced = df.sample(n=20000, random_state=42)  # random_state = reproducibility

# Save it back
df_reduced.to_csv("D:/MLOps-Student Score predictor/data/train_reduced.csv", index=False)

print(f"Reduced rows: {len(df_reduced)}")
