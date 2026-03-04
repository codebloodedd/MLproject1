import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_PATH = "data_clean/tmdb_clean.csv"
OUT_DIR = "outputs/module2/pca"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_COLS = ["popularity", "vote_average", "vote_count", "runtime", "budget", "revenue"]

def main():
    df = pd.read_csv(DATA_PATH)

    # --- Prepare PCA dataset (quant only, remove labels + nonnumeric) ---
    X = df[NUM_COLS].copy()
    X.to_csv(os.path.join(OUT_DIR, "pca_input_numeric_only.csv"), index=False)

    # --- Normalize (required) ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- PCA 2D ---
    pca2 = PCA(n_components=2, random_state=42)
    X2 = pca2.fit_transform(X_scaled)
    ev2 = pca2.explained_variance_ratio_.sum()

    # --- PCA 3D ---
    pca3 = PCA(n_components=3, random_state=42)
    X3 = pca3.fit_transform(X_scaled)
    ev3 = pca3.explained_variance_ratio_.sum()

    # --- How many components for 95% variance? ---
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.argmax(cum >= 0.95) + 1)

    # --- Save variance tables (for screenshot) ---
    pd.DataFrame({
        "component": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca_full.explained_variance_ratio_,
        "cumulative": cum
    }).to_csv(os.path.join(OUT_DIR, "pca_variance_table.csv"), index=False)

    # --- Save key summary text ---
    with open(os.path.join(OUT_DIR, "pca_summary.txt"), "w") as f:
        f.write(f"PCA 2D variance retained: {ev2:.4f} ({ev2*100:.2f}%)\n")
        f.write(f"PCA 3D variance retained: {ev3:.4f} ({ev3*100:.2f}%)\n")
        f.write(f"Components needed for >=95% variance: {n95}\n\n")
        f.write("Top 3 eigenvalues (explained_variance_):\n")
        top3 = pca_full.explained_variance_[:3]
        f.write(", ".join([f"{v:.6f}" for v in top3]) + "\n")

    # --- Plot 2D ---
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], s=20)
    plt.title(f"PCA (2D) — Variance retained: {ev2*100:.2f}%")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_2d.png"), dpi=200)
    plt.close()

    # --- Plot 3D ---
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=18)
    ax.set_title(f"PCA (3D) — Variance retained: {ev3*100:.2f}%")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_3d.png"), dpi=200)
    plt.close()

    # --- Plot cumulative variance ---
    plt.figure()
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o")
    plt.axhline(0.95, linestyle="--")
    plt.title("Cumulative Explained Variance (PCA)")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_cumulative.png"), dpi=200)
    plt.close()

    print("✅ PCA done. Outputs saved to:", OUT_DIR)
    print(f"2D variance: {ev2*100:.2f}% | 3D variance: {ev3*100:.2f}% | components for 95%: {n95}")

if __name__ == "__main__":
    main()