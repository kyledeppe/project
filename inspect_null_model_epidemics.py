import pandas as pd

import config

results_dir = config.RESULTS_DIR / "tables"

files = [
    "null_model_sir_individual_results.csv",
    "null_model_sir_summary.csv",
    "null_model_best_strategies.csv"
]

for filename in files:

    path = results_dir / filename

    print("\n" + "=" * 65)
    print(filename)
    print("=" * 65)

    df = pd.read_csv(path)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())