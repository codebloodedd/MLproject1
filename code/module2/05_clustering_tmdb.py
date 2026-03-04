import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram

DATA_PATH = "data_clean/tmdb_clean.csv"
OUT_DIR = "outputs/module2/clustering"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_COLS = ["popularity", "vote_average", "vote_count", "runtime", "budget", "revenue"]
LABEL_COL = "label_popular_top25"   # we color by this original label

def main():
    df = pd.read_csv(DATA_PATH)

    y_label = df[LABEL_COL].copy()     # save for coloring only
    X = df[NUM_COLS].copy()            # unlabeled numeric-only dataset

    # Save "before/after" snapshots
    X.to_csv(os.path.join(OUT_DIR, "clustering_input_numeric_only_unlabeled.csv"), index=False)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA to 3D (recommended for visualization)
    pca3 = PCA(n_components=3, random_state=42)
    X3 = pca3.fit_transform(X_scaled)
    var3 = pca3.explained_variance_ratio_.sum()

    with open(os.path.join(OUT_DIR, "clustering_prep_summary.txt"), "w") as f:
        f.write(f"Used label for coloring (removed from clustering): {LABEL_COL}\n")
        f.write(f"Features: {NUM_COLS}\n")
        f.write(f"PCA 3D variance retained: {var3:.4f} ({var3*100:.2f}%)\n")

    # We’ll use 2D PCA for plotting KMeans nicely
    pca2 = PCA(n_components=2, random_state=42)
    X2 = pca2.fit_transform(X_scaled)

    # -------- KMEANS + SILHOUETTE (pick 3 smart k values) --------
    scores = []
    k_range = range(2, 11)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        s = silhouette_score(X_scaled, labels)
        scores.append((k, s))

    scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)
    top3 = scores_sorted[:3]

    pd.DataFrame(scores, columns=["k", "silhouette"]).to_csv(
        os.path.join(OUT_DIR, "silhouette_scores.csv"), index=False
    )

    # Plot silhouette curve
    plt.figure()
    plt.plot([k for k, _ in scores], [s for _, s in scores], marker="o")
    plt.title("Silhouette Score by k (KMeans)")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "silhouette_curve.png"), dpi=200)
    plt.close()

    # For each chosen k: plot clusters + centroids (in PCA2 space) but color points by original label
    for idx, (k, s) in enumerate(top3, start=1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster = km.fit_predict(X_scaled)

        centroids_scaled = km.cluster_centers_
        centroids_2d = pca2.transform(centroids_scaled)

        plt.figure()
        # Color by original labels (requirement)
        plt.scatter(X2[:, 0], X2[:, 1], c=y_label, s=25)
        plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", s=200)
        plt.title(f"KMeans (k={k}) | silhouette={s:.3f} | color=original label ({LABEL_COL})")
        plt.xlabel("PCA2 - PC1")
        plt.ylabel("PCA2 - PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"kmeans_{idx}_k{k}.png"), dpi=200)
        plt.close()

    # -------- HIERARCHICAL (dendrogram) --------
    Z = linkage(X_scaled, method="ward")
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=30)
    plt.title("Hierarchical Clustering Dendrogram (Ward) — truncated")
    plt.xlabel("Cluster")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "hierarchical_dendrogram.png"), dpi=200)
    plt.close()

    # -------- DBSCAN --------
    # Reasonable starter eps; you can tune if needed
    db = DBSCAN(eps=1.2, min_samples=6)
    db_labels = db.fit_predict(X_scaled)

    # Plot DBSCAN in PCA2 space, color by cluster (noise = -1)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=db_labels, s=25)
    plt.title("DBSCAN clusters in PCA(2D) space (noise = -1)")
    plt.xlabel("PCA2 - PC1")
    plt.ylabel("PCA2 - PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dbscan.png"), dpi=200)
    plt.close()

    # Save basic DBSCAN stats
    n_noise = int((db_labels == -1).sum())
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

    with open(os.path.join(OUT_DIR, "dbscan_summary.txt"), "w") as f:
        f.write(f"DBSCAN eps=1.2 min_samples=6\n")
        f.write(f"Clusters found: {n_clusters}\n")
        f.write(f"Noise points: {n_noise}\n")

    print("✅ Clustering done. Outputs saved to:", OUT_DIR)
    print("Top 3 k by silhouette:", top3)

if __name__ == "__main__":
    main()