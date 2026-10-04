"""
Script to download TMDB 5000 movies dataset.
Run this once to prepare the data.
"""
import os
import urllib.request
import zipfile

# We'll use a bundled sample dataset if Kaggle is not available
SAMPLE_MOVIES_URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/tmdb_5000_movies.csv"

def download_sample():
    os.makedirs("data", exist_ok=True)
    print("Downloading sample movies dataset...")
    try:
        urllib.request.urlretrieve(
            SAMPLE_MOVIES_URL,
            "data/tmdb_5000_movies.csv"
        )
        print("Download complete!")
    except Exception as e:
        print(f"Could not download: {e}")

if __name__ == "__main__":
    download_sample()
