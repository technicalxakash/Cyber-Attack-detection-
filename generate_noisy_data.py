import numpy as np
import pandas as pd

# Load the original dataset
file_path = "top_10_features.csv"
data = pd.read_csv(file_path)

# Introduce random noise to numerical columns
np.random.seed(42)  # Set seed for consistent results
noise_factor = 0.2  # Amount of noise (increase to lower accuracy)

for col in data.columns:
    if col != "label":  # Don't modify the label column
        data[col] += np.random.normal(0, noise_factor * data[col].std(), size=data.shape[0])

# Shuffle 10% of the labels to make the dataset harder
num_shuffled = int(0.1 * len(data))  # 10% of the labels will be incorrect
shuffle_indices = np.random.choice(data.index, num_shuffled, replace=False)
data.loc[shuffle_indices, "label"] = data["label"].sample(frac=1).values[:num_shuffled]

# Save the noisy dataset
data.to_csv("noisy_top_10_features.csv", index=False)

print("✅ Noisy dataset saved as 'noisy_top_10_features.csv'")
