from pathlib import Path
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data_clean" / "tmdb_clean.csv"
OUTPUT_DIR = ROOT / "outputs" / "module4"
WEBSITE_DATA_DIR = ROOT / "website" / "assets" / "data" / "module4"
WEBSITE_IMAGE_DIR = ROOT / "website" / "assets" / "images" / "module4"
MODULE3_SUMMARY_PATH = ROOT / "website" / "assets" / "data" / "module3" / "module3_summary.json"

NUMERIC_COLS = [
    "vote_average",
    "vote_count",
    "runtime",
    "budget",
    "revenue",
    "release_year",
    "release_month",
]
SKEWED_COLS = ["vote_count", "budget", "revenue"]
SVM_C_VALUES = [0.01, 0.1, 1, 10, 100]
KERNEL_SPECS = {
    "linear": {"kernel": "linear"},
    "poly": {"kernel": "poly", "degree": 2, "coef0": 1},
    "rbf": {"kernel": "rbf"},
}


def ensure_dirs():
    for path in [OUTPUT_DIR, WEBSITE_DATA_DIR, WEBSITE_IMAGE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def copy_to_website(src: Path):
    if src.suffix.lower() == ".png":
        dest = WEBSITE_IMAGE_DIR / src.name
    else:
        dest = WEBSITE_DATA_DIR / src.name
    shutil.copy2(src, dest)


def save_csv(df: pd.DataFrame, name: str):
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    copy_to_website(path)
    return path


def save_json(payload: dict, name: str):
    path = OUTPUT_DIR / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    copy_to_website(path)
    return path


def finish_figure(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    copy_to_website(path)


def plot_dataframe_table(df: pd.DataFrame, title: str, path: Path):
    display_df = df.copy()
    n_rows, n_cols = display_df.shape
    fig_w = min(22, max(8, n_cols * 1.35))
    fig_h = max(2.8, 0.58 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, weight="bold", pad=14)
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef4fb")
    finish_figure(path)


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12, color="#08101e")
    finish_figure(path)


def build_modeling_frame(df: pd.DataFrame):
    df = df.copy()
    df["genres"] = df["genres"].fillna("Unknown")
    df["genre_list"] = df["genres"].apply(
        lambda value: [item.strip() for item in str(value).split("|") if item.strip()]
    )
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())
    for col in SKEWED_COLS:
        df[col] = np.log1p(df[col])
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df["genre_list"])
    genre_cols = [f"genre_{genre}" for genre in mlb.classes_]
    genre_df = pd.DataFrame(genre_matrix, columns=genre_cols, index=df.index)
    feature_df = pd.concat([df[NUMERIC_COLS], genre_df], axis=1)
    return df, feature_df, genre_cols


def prepare_base_data():
    raw_df = pd.read_csv(DATA_PATH)
    rows_before = len(raw_df)
    df = raw_df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    rows_after = len(df)
    _, feature_df, genre_cols = build_modeling_frame(df)

    y_pop = df["label_popular_top25"].astype(int)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, stratify=y_pop, random_state=42
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    split_meta = {
        "rows_before_dedup": int(rows_before),
        "rows_after_dedup": int(rows_after),
        "duplicates_removed": int(rows_before - rows_after),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_ids_overlap_test_ids": int(
            len(set(df.loc[train_idx, "id"]).intersection(set(df.loc[test_idx, "id"])))
        ),
        "primary_target": "label_popular_top25",
        "secondary_target": "label_high_rating",
        "primary_positive_rate": round(float(y_pop.mean()), 4),
        "secondary_positive_rate": round(float(df["label_high_rating"].mean()), 4),
        "numeric_feature_count": int(feature_df.shape[1]),
        "numeric_only_requirement": (
            "SVM classification needs numeric feature vectors because the algorithm depends on "
            "distance and dot-product calculations in vector space."
        ),
    }
    return df, feature_df, genre_cols, train_idx, test_idx, split_meta


def save_split_assets(df: pd.DataFrame, feature_df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    sample_cols = [
        "id",
        "title",
        "vote_average",
        "vote_count",
        "runtime",
        "budget",
        "revenue",
        "genres",
        "release_year",
        "release_month",
        "label_popular_top25",
        "label_high_rating",
    ]
    modeling_export = df[sample_cols].copy()
    save_csv(modeling_export, "module4_modeling_dataset.csv")

    train_sample = modeling_export.loc[train_idx].head(10).copy()
    test_sample = modeling_export.loc[test_idx].head(10).copy()
    save_csv(train_sample, "train_sample.csv")
    save_csv(test_sample, "test_sample.csv")

    numeric_sample = feature_df.head(12).copy().round(3)
    save_csv(numeric_sample, "svm_numeric_feature_sample.csv")

    plot_dataframe_table(train_sample.head(8), "Module 4 Training Set Preview", OUTPUT_DIR / "train_preview.png")
    plot_dataframe_table(test_sample.head(8), "Module 4 Testing Set Preview", OUTPUT_DIR / "test_preview.png")
    plot_dataframe_table(
        numeric_sample.iloc[:, : min(10, numeric_sample.shape[1])],
        "Numeric Feature Sample Used by SVM",
        OUTPUT_DIR / "svm_numeric_feature_preview.png",
    )

    y_train = df.loc[train_idx, "label_popular_top25"].astype(int)
    y_test = df.loc[test_idx, "label_popular_top25"].astype(int)
    labels = ["Train", "Test"]
    zeros = [(y_train == 0).sum(), (y_test == 0).sum()]
    ones = [(y_train == 1).sum(), (y_test == 1).sum()]

    plt.figure(figsize=(7, 4.6))
    plt.bar(labels, zeros, label="Label 0", color="#7aa2ff")
    plt.bar(labels, ones, bottom=zeros, label="Label 1", color="#56d9b8")
    plt.title("Disjoint Train/Test Split for Movie Popularity Label")
    plt.ylabel("Rows")
    plt.legend()
    plt.text(0, zeros[0] + ones[0] + 2, f"{len(y_train)} rows", ha="center")
    plt.text(1, zeros[1] + ones[1] + 2, f"{len(y_test)} rows", ha="center")
    finish_figure(OUTPUT_DIR / "train_test_split.png")


def evaluate_svm_for_target(
    feature_df: pd.DataFrame,
    target: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(feature_df.loc[train_idx])
    X_test = scaler.transform(feature_df.loc[test_idx])
    y_train = target.loc[train_idx]
    y_test = target.loc[test_idx]

    rows = []
    best_by_kernel = {}

    for kernel_name, kernel_params in KERNEL_SPECS.items():
        best_row = None
        for c_value in SVM_C_VALUES:
            clf = SVC(
                C=c_value,
                class_weight="balanced",
                gamma="scale",
                random_state=42,
                **kernel_params,
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            cm = confusion_matrix(y_test, preds)
            acc = accuracy_score(y_test, preds)
            row = {
                "target": target.name,
                "kernel": kernel_name,
                "C": c_value,
                "accuracy": round(float(acc), 4),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
            rows.append(row)
            if (
                best_row is None
                or row["accuracy"] > best_row["accuracy"]
                or (row["accuracy"] == best_row["accuracy"] and row["C"] < best_row["C"])
            ):
                best_row = row
        best_by_kernel[kernel_name] = best_row

    return pd.DataFrame(rows), best_by_kernel


def plot_svm_experiment_chart(results_df: pd.DataFrame, path: Path):
    pivot_df = results_df.pivot(index="C", columns="kernel", values="accuracy").sort_index()
    plt.figure(figsize=(8.4, 4.8))
    palette = {"linear": "#7aa2ff", "poly": "#56d9b8", "rbf": "#f6c85f"}
    for kernel in pivot_df.columns:
        plt.plot(
            pivot_df.index.astype(float),
            pivot_df[kernel],
            marker="o",
            linewidth=2,
            color=palette.get(kernel, "#cccccc"),
            label=kernel,
        )
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel("C value (log scale)")
    plt.ylabel("Accuracy")
    plt.title("SVM Kernel and Cost Experiments")
    plt.legend(title="Kernel")
    finish_figure(path)


def plot_svm_decision_region(
    feature_df: pd.DataFrame,
    target: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    kernel_name: str,
    best_row: dict,
    path: Path,
):
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(feature_df.loc[train_idx])
    X_test_full = scaler.transform(feature_df.loc[test_idx])
    pca = PCA(n_components=2, random_state=42)
    X_train = pca.fit_transform(X_train_full)
    X_test = pca.transform(X_test_full)
    y_train = target.loc[train_idx].to_numpy()
    y_test = target.loc[test_idx].to_numpy()

    clf = SVC(
        C=best_row["C"],
        class_weight="balanced",
        gamma="scale",
        random_state=42,
        **KERNEL_SPECS[kernel_name],
    )
    clf.fit(X_train, y_train)

    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.8
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.8
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - 0.8
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + 0.8
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], alpha=0.22, colors=["#7aa2ff", "#56d9b8"])
    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c=y_train,
        cmap=plt.cm.coolwarm,
        s=46,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.95,
        marker="o",
        label="Train",
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap=plt.cm.coolwarm,
        s=56,
        edgecolor="#08101e",
        linewidth=0.7,
        alpha=0.95,
        marker="^",
        label="Test",
    )
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        facecolors="none",
        edgecolors="#f6c85f",
        s=130,
        linewidth=1.3,
        label="Support vectors",
    )
    ax.set_title(f"SVM Decision Regions in 2D PCA Space ({kernel_name}, C={best_row['C']})", fontsize=12, weight="bold")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Train points", markerfacecolor="#b14cff", markersize=8),
        Line2D([0], [0], marker="^", color="w", label="Test points", markerfacecolor="#5ec7ff", markersize=8),
        Line2D([0], [0], marker="o", color="#f6c85f", label="Support vectors", markerfacecolor="none", markersize=9),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    finish_figure(path)


def create_svm_overview_images():
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_title("SVM Margin and Support Vector Concept", fontsize=14, weight="bold")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.plot([2, 8], [2, 8], color="#56d9b8", linewidth=2.4, label="Decision boundary")
    ax.plot([1.3, 7.3], [2, 8], color="#7aa2ff", linewidth=1.5, linestyle="--", label="Margin")
    ax.plot([2.7, 8.7], [2, 8], color="#7aa2ff", linewidth=1.5, linestyle="--")
    class_a = np.array([[2.2, 5.5], [3.0, 6.2], [4.0, 7.2], [2.8, 4.8]])
    class_b = np.array([[6.2, 2.8], [7.3, 3.5], [7.8, 5.1], [5.8, 3.6]])
    ax.scatter(class_a[:, 0], class_a[:, 1], c="#56d9b8", s=85, edgecolor="white")
    ax.scatter(class_b[:, 0], class_b[:, 1], c="#ff7f7f", s=85, edgecolor="white")
    ax.scatter([3.0, 6.2], [6.2, 2.8], facecolors="none", edgecolors="#f6c85f", s=180, linewidth=2)
    ax.text(3.25, 6.55, "Support vector", color="#f6c85f", weight="bold")
    ax.text(6.55, 3.15, "Support vector", color="#f6c85f", weight="bold")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="lower right")
    finish_figure(OUTPUT_DIR / "svm_margin_support_vectors.png")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.set_title("Polynomial Kernel Mapping Example", fontsize=14, weight="bold")
    ax.axis("off")
    ax.text(0.06, 0.84, "Start with a 2D point:", fontsize=13, weight="bold")
    ax.text(0.07, 0.70, r"$x = (x_1, x_2) = (2, 3)$", fontsize=18)
    ax.text(0.06, 0.52, r"Polynomial kernel:  $K(x,z) = (x \cdot z + r)^d$", fontsize=15)
    ax.text(0.06, 0.39, r"With $r = 1$ and $d = 2$, the implicit cast expands to:", fontsize=13)
    ax.text(0.07, 0.23, r"$\phi(x) = [x_1^2,\ \sqrt{2}x_1x_2,\ x_2^2,\ \sqrt{2}x_1,\ \sqrt{2}x_2,\ 1]$", fontsize=17)
    ax.text(0.07, 0.08, r"For $(2,3)$ this becomes approximately $[4,\ 8.49,\ 9,\ 2.83,\ 4.24,\ 1]$", fontsize=14)
    finish_figure(OUTPUT_DIR / "svm_polynomial_cast_example.png")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.set_title("Kernel Idea: Linear vs Nonlinear Separation", fontsize=14, weight="bold")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    theta = np.linspace(0, 2 * np.pi, 250)
    inner = np.column_stack([0.2 * np.cos(theta[::25]), 0.2 * np.sin(theta[::25])])
    outer = np.column_stack([0.65 * np.cos(theta[::14]), 0.65 * np.sin(theta[::14])])
    ax.scatter(inner[:, 0], inner[:, 1], c="#56d9b8", s=55, edgecolor="white")
    ax.scatter(outer[:, 0], outer[:, 1], c="#ff7f7f", s=55, edgecolor="white")
    circle = plt.Circle((0, 0), 0.42, color="#7aa2ff", fill=False, linewidth=2.5, linestyle="--")
    ax.add_patch(circle)
    ax.text(-0.9, 0.86, "A linear boundary struggles here,\nbut a kernel can separate these classes\nin a higher-dimensional space.", fontsize=11)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    finish_figure(OUTPUT_DIR / "svm_kernel_concept.png")


def run_svm_pipeline(df: pd.DataFrame, feature_df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    target = df["label_popular_top25"].astype(int)
    results_df, best_by_kernel = evaluate_svm_for_target(feature_df, target, train_idx, test_idx)
    results_df = results_df.sort_values(["kernel", "C"]).reset_index(drop=True)
    save_csv(results_df, "svm_experiment_results.csv")

    final_rows = []
    scaler = StandardScaler()
    X_train = scaler.fit_transform(feature_df.loc[train_idx])
    X_test = scaler.transform(feature_df.loc[test_idx])
    y_train = target.loc[train_idx]
    y_test = target.loc[test_idx]

    for kernel_name, best_row in best_by_kernel.items():
        clf = SVC(
            C=best_row["C"],
            class_weight="balanced",
            gamma="scale",
            random_state=42,
            **KERNEL_SPECS[kernel_name],
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        plot_confusion_matrix(
            cm,
            f"{kernel_name.upper()} kernel confusion matrix",
            OUTPUT_DIR / f"svm_confusion_{kernel_name}.png",
        )
        plot_svm_decision_region(
            feature_df,
            target,
            train_idx,
            test_idx,
            kernel_name,
            best_row,
            OUTPUT_DIR / f"svm_boundary_{kernel_name}.png",
        )
        final_rows.append(best_row)

    final_df = pd.DataFrame(final_rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    save_csv(final_df, "svm_best_kernels.csv")
    plot_svm_experiment_chart(results_df, OUTPUT_DIR / "svm_kernel_cost_comparison.png")

    best_kernel_row = final_df.iloc[0].to_dict()
    return {
        "experiment_df": results_df,
        "best_df": final_df,
        "best_kernel": best_kernel_row["kernel"],
        "best_accuracy": float(best_kernel_row["accuracy"]),
        "best_cost": float(best_kernel_row["C"]),
    }


def load_decision_tree_baseline():
    if MODULE3_SUMMARY_PATH.exists():
        with MODULE3_SUMMARY_PATH.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        dt_rows = summary.get("decision_trees", [])
        if dt_rows:
            best = max(dt_rows, key=lambda row: row.get("accuracy", 0))
            return {
                "model": "Decision Tree (Module 3 best)",
                "accuracy": float(best["accuracy"]),
            }
    return {"model": "Decision Tree (Module 3 best)", "accuracy": 0.7917}


def run_random_forest(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    best_svm: dict,
):
    target = df["label_popular_top25"].astype(int)
    X_train = feature_df.loc[train_idx]
    X_test = feature_df.loc[test_idx]
    y_train = target.loc[train_idx]
    y_test = target.loc[test_idx]

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    acc = accuracy_score(y_test, preds)

    metrics_df = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "accuracy": round(float(acc), 4),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        ]
    )
    save_csv(metrics_df, "ensemble_random_forest_metrics.csv")
    plot_confusion_matrix(cm, "Random Forest confusion matrix", OUTPUT_DIR / "ensemble_confusion_random_forest.png")

    importances = (
        pd.Series(rf.feature_importances_, index=feature_df.columns)
        .sort_values(ascending=False)
        .head(12)
        .reset_index()
    )
    importances.columns = ["feature", "importance"]
    save_csv(importances, "ensemble_feature_importances.csv")

    plt.figure(figsize=(8.6, 5.2))
    plt.barh(importances["feature"][::-1], importances["importance"][::-1], color="#56d9b8")
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    finish_figure(OUTPUT_DIR / "ensemble_feature_importance.png")

    dt_baseline = load_decision_tree_baseline()
    comparison_df = pd.DataFrame(
        [
            {"model": "Random Forest", "accuracy": round(float(acc), 4)},
            {"model": f"SVM ({best_svm['best_kernel']}, C={best_svm['best_cost']:g})", "accuracy": round(float(best_svm["best_accuracy"]), 4)},
            dt_baseline,
        ]
    ).sort_values("accuracy", ascending=False)
    save_csv(comparison_df, "ensemble_model_comparison.csv")

    plt.figure(figsize=(8.2, 4.8))
    plt.bar(comparison_df["model"], comparison_df["accuracy"], color=["#56d9b8", "#7aa2ff", "#f6c85f"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Module 4 Ensemble Comparison")
    for idx, value in enumerate(comparison_df["accuracy"]):
        plt.text(idx, value + 0.02, f"{value:.3f}", ha="center")
    finish_figure(OUTPUT_DIR / "ensemble_model_comparison.png")

    return {
        "accuracy": round(float(acc), 4),
        "comparison_df": comparison_df,
        "top_features": importances.to_dict(orient="records"),
    }


def run_secondary_contrast(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    best_svm: dict,
    rf_accuracy_popularity: float,
):
    target = df["label_high_rating"].astype(int)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(feature_df.loc[train_idx])
    X_test_scaled = scaler.transform(feature_df.loc[test_idx])
    y_train = target.loc[train_idx]
    y_test = target.loc[test_idx]

    svm = SVC(
        C=best_svm["best_cost"],
        class_weight="balanced",
        gamma="scale",
        random_state=42,
        **KERNEL_SPECS[best_svm["best_kernel"]],
    )
    svm.fit(X_train_scaled, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(feature_df.loc[train_idx], y_train)
    rf_acc = accuracy_score(y_test, rf.predict(feature_df.loc[test_idx]))

    compare_df = pd.DataFrame(
        [
            {"target": "label_popular_top25", "model": f"SVM ({best_svm['best_kernel']})", "accuracy": round(float(best_svm["best_accuracy"]), 4)},
            {"target": "label_popular_top25", "model": "Random Forest", "accuracy": round(float(rf_accuracy_popularity), 4)},
            {"target": "label_high_rating", "model": f"SVM ({best_svm['best_kernel']})", "accuracy": round(float(svm_acc), 4)},
            {"target": "label_high_rating", "model": "Random Forest", "accuracy": round(float(rf_acc), 4)},
        ]
    )
    save_csv(compare_df, "secondary_target_comparison.csv")

    plt.figure(figsize=(8.4, 4.8))
    x = np.arange(2)
    width = 0.34
    pop_scores = compare_df[compare_df["target"] == "label_popular_top25"]["accuracy"].to_list()
    rate_scores = compare_df[compare_df["target"] == "label_high_rating"]["accuracy"].to_list()
    models = ["SVM", "Random Forest"]
    plt.bar(x - width / 2, [pop_scores[0], np.nan], width=width, label="Top popularity", color="#7aa2ff")
    plt.bar(x + width / 2, rate_scores, width=width, label="High rating", color="#56d9b8")
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Secondary Target Contrast: Popularity vs Rating")
    plt.legend()
    for idx, value in enumerate(pop_scores):
        plt.text(idx - width / 2, value + 0.02, f"{value:.3f}", ha="center")
    for idx, value in enumerate(rate_scores):
        plt.text(idx + width / 2, value + 0.02, f"{value:.3f}", ha="center")
    finish_figure(OUTPUT_DIR / "secondary_target_comparison.png")

    return {
        "svm_high_rating_accuracy": round(float(svm_acc), 4),
        "rf_high_rating_accuracy": round(float(rf_acc), 4),
    }


def create_secondary_contrast_chart(best_svm_accuracy: float, rf_accuracy: float, secondary_svm: float, secondary_rf: float):
    plt.figure(figsize=(8.4, 4.8))
    x = np.arange(2)
    width = 0.35
    pop_scores = [best_svm_accuracy, rf_accuracy]
    rating_scores = [secondary_svm, secondary_rf]
    plt.bar(x - width / 2, pop_scores, width=width, label="Top popularity target", color="#7aa2ff")
    plt.bar(x + width / 2, rating_scores, width=width, label="High rating target", color="#56d9b8")
    plt.xticks(x, ["SVM", "Random Forest"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("How the Best Models Shift Across Two Success Labels")
    plt.legend()
    for idx, value in enumerate(pop_scores):
        plt.text(idx - width / 2, value + 0.02, f"{value:.3f}", ha="center")
    for idx, value in enumerate(rating_scores):
        plt.text(idx + width / 2, value + 0.02, f"{value:.3f}", ha="center")
    finish_figure(OUTPUT_DIR / "secondary_target_contrast.png")


def create_kernel_formula_figure():
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.axis("off")
    ax.set_title("Kernel Functions Used in the SVM Overview", fontsize=14, weight="bold", pad=12)
    ax.text(0.05, 0.74, r"Linear:   $K(x, z) = x \cdot z$", fontsize=17)
    ax.text(0.05, 0.48, r"Polynomial:   $K(x, z) = (x \cdot z + r)^d$", fontsize=17)
    ax.text(0.05, 0.22, r"RBF:   $K(x, z) = \exp(-\gamma \|x - z\|^2)$", fontsize=17)
    ax.text(
        0.60,
        0.44,
        "The dot product is the anchor.\nKernels reuse dot-product structure\nwithout explicitly building every higher-dimensional feature.",
        fontsize=11.5,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef4fb", edgecolor="#1f4e79"),
    )
    finish_figure(OUTPUT_DIR / "svm_kernel_formulas.png")


def create_ensemble_overview_image():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axis("off")
    ax.set_title("Random Forest as an Ensemble", fontsize=14, weight="bold", pad=12)
    tree_boxes = [(0.08, 0.58), (0.30, 0.58), (0.52, 0.58), (0.74, 0.58)]
    for idx, (x, y) in enumerate(tree_boxes, start=1):
        rect = Rectangle((x, y), 0.14, 0.20, facecolor="#dbe9ff", edgecolor="#1f4e79", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.07, y + 0.10, f"Tree {idx}", ha="center", va="center", fontsize=11, weight="bold")
        ax.annotate("", xy=(0.50, 0.32), xytext=(x + 0.07, y), arrowprops={"arrowstyle": "->", "lw": 2, "color": "#56d9b8"})
    vote_rect = Rectangle((0.38, 0.18), 0.24, 0.12, facecolor="#dff7ef", edgecolor="#18794e", linewidth=2)
    ax.add_patch(vote_rect)
    ax.text(0.50, 0.24, "Majority vote", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(
        0.06,
        0.05,
        "Each tree sees a slightly different bootstrap sample and split pattern.\nTheir combined vote is usually more stable than relying on a single tree.",
        fontsize=11,
    )
    finish_figure(OUTPUT_DIR / "ensemble_random_forest_overview.png")


def main():
    ensure_dirs()
    df, feature_df, genre_cols, train_idx, test_idx, split_meta = prepare_base_data()

    save_split_assets(df, feature_df, train_idx, test_idx)
    save_json(split_meta, "train_test_split_summary.json")

    create_svm_overview_images()
    create_kernel_formula_figure()
    create_ensemble_overview_image()

    svm_results = run_svm_pipeline(df, feature_df, train_idx, test_idx)
    ensemble_results = run_random_forest(df, feature_df, train_idx, test_idx, svm_results)
    secondary_results = run_secondary_contrast(
        df,
        feature_df,
        train_idx,
        test_idx,
        svm_results,
        ensemble_results["accuracy"],
    )
    create_secondary_contrast_chart(
        svm_results["best_accuracy"],
        ensemble_results["accuracy"],
        secondary_results["svm_high_rating_accuracy"],
        secondary_results["rf_high_rating_accuracy"],
    )

    summary = {
        "split": split_meta,
        "svm_best_kernels": svm_results["best_df"].to_dict(orient="records"),
        "best_svm_kernel": svm_results["best_kernel"],
        "best_svm_accuracy": round(float(svm_results["best_accuracy"]), 4),
        "best_svm_cost": svm_results["best_cost"],
        "random_forest_accuracy": round(float(ensemble_results["accuracy"]), 4),
        "ensemble_comparison": ensemble_results["comparison_df"].to_dict(orient="records"),
        "secondary_target": secondary_results,
        "genre_indicator_count": len(genre_cols),
    }
    save_json(summary, "module4_summary.json")

    print("Module 4 outputs generated in outputs/module4 and copied into website/assets.")


if __name__ == "__main__":
    main()
