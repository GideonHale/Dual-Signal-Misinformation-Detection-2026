import pandas as pd

# 1. Load the separate ISOT files
true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')

# 2. Add the binary ground truth labels
true_df['label'] = 1
fake_df['label'] = 0

# 3. Combine them into a single dataset
isot_combined = pd.concat([true_df, fake_df], ignore_index=True)

# 4. Optional but recommended: Shuffle the dataset so true/false aren't clumped together
isot_combined = isot_combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save your new, clean ground truth file!
isot_combined.to_csv('ISOT_combined_labeled.csv', index=False)
print(f"Combined dataset created with {len(isot_combined)} total articles.")