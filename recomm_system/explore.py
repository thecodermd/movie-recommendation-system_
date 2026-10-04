"""
Notebook-style analysis script.
Run this to explore the dataset interactively.
Usage: python explore.py
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

DATA = "data/tmdb_5000_movies.csv"
if not os.path.exists(DATA):
    print("Dataset not found! Run: python setup.py")
    sys.exit(1)

df = pd.read_csv(DATA)
print("=" * 60)
print("TMDB 5000 Movies - Dataset Overview")
print("=" * 60)
print(f"Shape     : {df.shape}")
print(f"Columns   : {list(df.columns)}")
print(f"\nSample:\n{df[['title','vote_average','genres']].head()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nRating stats:\n{df['vote_average'].describe()}")
