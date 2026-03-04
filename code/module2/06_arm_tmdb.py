import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import networkx as nx

DATA_PATH = "data_clean/tmdb_clean.csv"
OUT_DIR = "outputs/module2/arm"
os.makedirs(OUT_DIR, exist_ok=True)

def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    # Simple bins that make ARM work well
    df = df.copy()

    # Budget / revenue skew: use median split (stable for small datasets)
    df["BudgetBin"] = np.where(df["budget"] >= df["budget"].median(), "HighBudget", "LowBudget")
    df["RevenueBin"] = np.where(df["revenue"] >= df["revenue"].median(), "HighRevenue", "LowRevenue")
    df["RuntimeBin"] = np.where(df["runtime"] >= df["runtime"].median(), "LongRuntime", "ShortRuntime")

    # Rating threshold aligned with your label definition
    df["RatingBin"] = np.where(df["vote_average"] >= 7.0, "HighRating", "LowRating")
    df["PopularityBin"] = np.where(df["label_popular_top25"] == 1, "PopularTop25", "NotPopularTop25")

    # Release seasonality
    df["ReleaseSeason"] = np.where(df["release_month"].isin([6,7,8]), "SummerRelease", "NonSummerRelease")

    return df

def build_transactions(df: pd.DataFrame) -> list[list[str]]:
    tx = []
    for _, row in df.iterrows():
        items = []

        # Genres become separate items
        if isinstance(row["genres"], str) and row["genres"].strip():
            items.extend([g.strip() for g in row["genres"].split("|") if g.strip()])

        # Add discretized numeric bins
        items.extend([
            row["BudgetBin"],
            row["RevenueBin"],
            row["RuntimeBin"],
            row["RatingBin"],
            row["PopularityBin"],
            row["ReleaseSeason"],
        ])

        tx.append(sorted(set(items)))
    return tx

def main():
    df = pd.read_csv(DATA_PATH)

    # Prepare and build transaction dataset
    df2 = bucketize(df)
    transactions = build_transactions(df2)

    # Save a sample of transactions (requirement: show + link)
    sample_path = os.path.join(OUT_DIR, "arm_transactions_sample.csv")
    pd.DataFrame({"transaction": ["; ".join(t) for t in transactions]}).head(30).to_csv(sample_path, index=False)

    # One-hot encode transactions
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    onehot = pd.DataFrame(te_ary, columns=te.columns_)

    onehot.to_csv(os.path.join(OUT_DIR, "arm_onehot_matrix.csv"), index=False)

    # Apriori
    # For 218 rows, start with min_support 0.07; adjust if rules are too few/many
    min_support = 0.07
    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    freq.to_csv(os.path.join(OUT_DIR, "frequent_itemsets.csv"), index=False)

    # Rules
    rules = association_rules(freq, metric="confidence", min_threshold=0.3)
    rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    rules.to_csv(os.path.join(OUT_DIR, "all_rules.csv"), index=False)

    # Top 15 by each metric
    top_support = rules.sort_values("support", ascending=False).head(15)
    top_conf = rules.sort_values("confidence", ascending=False).head(15)
    top_lift = rules.sort_values("lift", ascending=False).head(15)

    top_support.to_csv(os.path.join(OUT_DIR, "top15_support.csv"), index=False)
    top_conf.to_csv(os.path.join(OUT_DIR, "top15_confidence.csv"), index=False)
    top_lift.to_csv(os.path.join(OUT_DIR, "top15_lift.csv"), index=False)

    # -------- Network visualization (required) --------
    # Use top 15 lift rules to keep graph readable
    G = nx.DiGraph()
    for _, r in top_lift.iterrows():
        lhs = ", ".join(sorted(list(r["antecedents"])))
        rhs = ", ".join(sorted(list(r["consequents"])))
        G.add_edge(lhs, rhs, weight=float(r["lift"]))

    plt.figure(figsize=(12, 7))
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=2200, font_size=8, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}, font_size=7)
    plt.title("ARM Rule Network (Top 15 by Lift)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "arm_network_top15_lift.png"), dpi=200)
    plt.close()

    # Summary file (thresholds)
    with open(os.path.join(OUT_DIR, "arm_summary.txt"), "w") as f:
        f.write(f"min_support used: {min_support}\n")
        f.write("confidence threshold: 0.3\n")
        f.write(f"total rules generated: {len(rules)}\n")

    print("✅ ARM done. Outputs saved to:", OUT_DIR)
    print("Rules generated:", len(rules))

if __name__ == "__main__":
    main()