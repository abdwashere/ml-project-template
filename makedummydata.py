"""
Generates fake classification data so you can smoke-test the
template's wiring before plugging in a real dataset.

Run once: python make_dummy_data.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

n_samples = 200
df = pd.DataFrame({
    "feature_1": np.random.randn(n_samples),
    "feature_2": np.random.randn(n_samples),
    "feature_3": np.random.randn(n_samples),
})
# Fake label, loosely correlated with feature_1 so the model has something to learn
df["label"] = (df["feature_1"] + np.random.randn(n_samples) * 0.5 > 0).astype(int)

os.makedirs("data", exist_ok=True)
df.to_csv("data/train.csv", index=False)
print(f"Created data/train.csv with {len(df)} rows")
print(df.head())