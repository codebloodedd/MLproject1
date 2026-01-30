import pandas as pd
import numpy as np

RAW = "data_raw/tmdb_enriched_raw.csv"
OUT = "data_clean/tmdb_clean.csv"

def safe_fill(df, col, value):
    """Fill missing values only if column exists."""
    if col in df.columns:
        df[col] = df[col].fillna(value)

def safe_numeric(df, col):
    """Convert to numeric only if column exists."""
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

def main():
    df = pd.read_csv(RAW)

    print("Columns found:")
    print(df.columns.tolist())

    # Keep only columns that exist
    wanted = [
        "id","title","release_date","release_date_full",
        "popularity","vote_average","vote_count","adult",
        "runtime","budget","revenue","genres","original_language"
    ]

    keep = [c for c in wanted if c in df.columns]
    df = df[keep].copy()

    # Convert numeric columns safely
    for c in ["popularity","vote_average","vote_count","runtime","budget","revenue"]:
        safe_numeric(df, c)

    # Parse dates if present
    if "release_date_full" in df.columns:
        df["release_date_full"] = pd.to_datetime(
            df["release_date_full"], errors="coerce"
        )
        df["release_year"] = df["release_date_full"].dt.year
        df["release_month"] = df["release_date_full"].dt.month

    # Fill missing values
    for c in ["runtime","budget","revenue","vote_average","vote_count","popularity"]:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    safe_fill(df, "genres", "Unknown")
    safe_fill(df, "original_language", "unknown")

    # Clean text
    for c in ["genres","original_language","title"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Cap extreme outliers (99th percentile)
    for c in ["budget","revenue"]:
        if c in df.columns:
            cap = df[c].quantile(0.99)
            df[c] = np.minimum(df[c], cap)

    # Create labels (only if popularity exists)
    if "popularity" in df.columns:
        pop_thresh = df["popularity"].quantile(0.75)
        df["label_popular_top25"] = (
            df["popularity"] >= pop_thresh
        ).astype(int)

    if "vote_average" in df.columns:
        df["label_high_rating"] = (
            df["vote_average"] >= 7.0
        ).astype(int)

    # Final check
    print("\nMissing values per column:")
    print(df.isna().sum())

    df.to_csv(OUT, index=False)

    print("\nSaved clean dataset:", OUT)
    print("\nFinal dtypes:")
    print(df.dtypes)

    print("\nSample rows:")
    print(df.head(5))

    print("\nFinal shape:", df.shape)


if __name__ == "__main__":
    main()
