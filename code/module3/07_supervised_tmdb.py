from pathlib import Path
import json
import math
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, MultiLabelBinarizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data_clean" / "tmdb_clean.csv"
OUTPUT_DIR = ROOT / "outputs" / "module3"
WEBSITE_DATA_DIR = ROOT / "website" / "assets" / "data" / "module3"
WEBSITE_IMAGE_DIR = ROOT / "website" / "assets" / "images" / "module3"


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
    fig_w = min(22, max(8, n_cols * 1.4))
    fig_h = max(2.8, 0.6 * (n_rows + 2))
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


def plot_confusion(ax, cm: np.ndarray, title: str):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="#0b1220", fontsize=11)


def plot_split_overview(y_train: pd.Series, y_test: pd.Series, path: Path):
    labels = ["Train", "Test"]
    zeros = [(y_train == 0).sum(), (y_test == 0).sum()]
    ones = [(y_train == 1).sum(), (y_test == 1).sum()]

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, zeros, label="Label 0", color="#7aa2ff")
    plt.bar(labels, ones, bottom=zeros, label="Label 1", color="#56d9b8")
    plt.title("Disjoint Train/Test Split for label_popular_top25")
    plt.ylabel("Rows")
    plt.legend()
    plt.text(0, zeros[0] + ones[0] + 2, f"{len(y_train)} rows", ha="center")
    plt.text(1, zeros[1] + ones[1] + 2, f"{len(y_test)} rows", ha="center")
    finish_figure(path)


def plot_nb_accuracy(results_df: pd.DataFrame, path: Path):
    plt.figure(figsize=(7, 4.5))
    plt.bar(results_df["model"], results_df["accuracy"], color=["#7aa2ff", "#56d9b8", "#f6c85f"])
    plt.ylim(0, 1)
    plt.title("Naive Bayes Accuracy Comparison")
    plt.ylabel("Accuracy")
    for idx, value in enumerate(results_df["accuracy"]):
        plt.text(idx, value + 0.02, f"{value:.3f}", ha="center")
    finish_figure(path)


def plot_regression_compare(results_df: pd.DataFrame, path: Path):
    plt.figure(figsize=(7.5, 4.5))
    plt.bar(results_df["model"], results_df["accuracy"], color=["#56d9b8", "#7aa2ff", "#f6c85f"])
    plt.ylim(0, 1)
    plt.title("Module 3 Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    for idx, value in enumerate(results_df["accuracy"]):
        plt.text(idx, value + 0.02, f"{value:.3f}", ha="center")
    finish_figure(path)


def plot_dt_overview_example(path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.set_title("Decision Tree Split Example", fontsize=15, weight="bold", pad=12)

    parent = Rectangle((0.05, 0.45), 0.24, 0.22, facecolor="#dbe9ff", edgecolor="#1f4e79", linewidth=2)
    left = Rectangle((0.42, 0.68), 0.22, 0.18, facecolor="#dff7ef", edgecolor="#18794e", linewidth=2)
    right = Rectangle((0.42, 0.18), 0.22, 0.18, facecolor="#ffe6d5", edgecolor="#b54708", linewidth=2)
    ax.add_patch(parent)
    ax.add_patch(left)
    ax.add_patch(right)

    ax.text(0.17, 0.56, "Parent node\n100 movies\n60 class 0 / 40 class 1", ha="center", va="center", fontsize=11)
    ax.text(0.53, 0.77, "Left child\n70 movies\n55 / 15", ha="center", va="center", fontsize=11)
    ax.text(0.53, 0.27, "Right child\n30 movies\n5 / 25", ha="center", va="center", fontsize=11)
    ax.annotate("", xy=(0.42, 0.75), xytext=(0.29, 0.60), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.annotate("", xy=(0.42, 0.28), xytext=(0.29, 0.52), arrowprops={"arrowstyle": "->", "lw": 2})
    ax.text(0.33, 0.67, "Split on\nvote_count > threshold", fontsize=10, weight="bold")
    ax.text(0.70, 0.74, "A good split makes\nchild nodes purer.", fontsize=11)
    finish_figure(path)


def plot_dt_impurity_example(path: Path):
    labels = ["Parent", "Left child", "Right child", "Weighted after split"]
    gini_values = [0.48, 0.337, 0.278, 0.319]
    entropy_values = [0.971, 0.750, 0.650, 0.720]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 4.8))
    plt.bar(x - width / 2, gini_values, width=width, label="Gini", color="#7aa2ff")
    plt.bar(x + width / 2, entropy_values, width=width, label="Entropy", color="#56d9b8")
    plt.xticks(x, labels)
    plt.ylabel("Impurity")
    plt.title("Impurity Drops After a Strong Split")
    plt.legend()
    finish_figure(path)


def prepare_base_data():
    df = pd.read_csv(DATA_PATH)
    before_rows = len(df)
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    after_rows = len(df)

    df["genres"] = df["genres"].fillna("Unknown")
    df["genre_list"] = df["genres"].apply(lambda value: [item.strip() for item in str(value).split("|") if item.strip()])

    numeric_cols = ["vote_average", "vote_count", "runtime", "budget", "revenue", "release_year", "release_month"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col].median())

    y = df["label_popular_top25"].astype(int)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.25, stratify=y, random_state=42)
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)

    split_meta = {
        "rows_before_dedup": int(before_rows),
        "rows_after_dedup": int(after_rows),
        "duplicates_removed": int(before_rows - after_rows),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_ids_overlap_test_ids": int(
            len(set(df.loc[train_idx, "id"]).intersection(set(df.loc[test_idx, "id"])))
        ),
        "target_name": "label_popular_top25",
        "target_rate": round(float(y.mean()), 4),
    }

    return df, train_idx, test_idx, split_meta


def build_multilabel_frame(df: pd.DataFrame):
    mlb = MultiLabelBinarizer()
    genre_df = pd.DataFrame(mlb.fit_transform(df["genre_list"]), columns=[f"genre_{c}" for c in mlb.classes_])
    genre_df.index = df.index
    return pd.concat([df, genre_df], axis=1), list(genre_df.columns)


def run_naive_bayes(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    y = df["label_popular_top25"].astype(int)
    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    genre_text_vectorizer = CountVectorizer(token_pattern=r"[^|]+")
    X_train_text = genre_text_vectorizer.fit_transform(train_df["genres"])
    X_test_text = genre_text_vectorizer.transform(test_df["genres"])
    genre_terms = [f"genre_{term}" for term in genre_text_vectorizer.get_feature_names_out()]

    mnb_numeric_cols = ["vote_average", "runtime", "vote_count", "budget", "revenue", "release_year", "release_month"]
    X_train_mnb_numeric = train_df[mnb_numeric_cols].copy()
    X_test_mnb_numeric = test_df[mnb_numeric_cols].copy()
    for col in ["vote_count", "budget", "revenue"]:
        X_train_mnb_numeric[col] = np.log1p(X_train_mnb_numeric[col])
        X_test_mnb_numeric[col] = np.log1p(X_test_mnb_numeric[col])
    X_train_mnb = np.hstack([X_train_mnb_numeric.to_numpy(), X_train_text.toarray()])
    X_test_mnb = np.hstack([X_test_mnb_numeric.to_numpy(), X_test_text.toarray()])
    mnb_feature_names = mnb_numeric_cols + genre_terms

    gnb_numeric_cols = ["vote_average", "runtime", "vote_count", "budget", "revenue", "release_year", "release_month"]
    gnb_genre_df, gnb_genre_cols = build_multilabel_frame(df[["genres", "genre_list"]].join(df[gnb_numeric_cols + ["label_popular_top25"]]))
    gnb_features = pd.concat([gnb_genre_df[gnb_numeric_cols], gnb_genre_df[gnb_genre_cols]], axis=1)
    scaler = StandardScaler()
    X_train_gnb = scaler.fit_transform(gnb_features.loc[train_idx])
    X_test_gnb = scaler.transform(gnb_features.loc[test_idx])

    bnb_df = df.copy()
    bnb_df["high_vote_average"] = (bnb_df["vote_average"] >= bnb_df["vote_average"].median()).astype(int)
    bnb_df["high_runtime"] = (bnb_df["runtime"] >= bnb_df["runtime"].median()).astype(int)
    bnb_df["high_vote_count"] = (bnb_df["vote_count"] >= bnb_df["vote_count"].median()).astype(int)
    bnb_df["high_budget"] = (bnb_df["budget"] >= bnb_df["budget"].median()).astype(int)
    bnb_df["high_revenue"] = (bnb_df["revenue"] >= bnb_df["revenue"].median()).astype(int)
    bnb_df["summer_release"] = bnb_df["release_month"].isin([5, 6, 7, 8]).astype(int)
    bnb_df["recent_release"] = (bnb_df["release_year"] >= bnb_df["release_year"].median()).astype(int)
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(bnb_df["genre_list"])
    genre_cols = [f"genre_{genre}" for genre in mlb.classes_]
    genre_binary_df = pd.DataFrame(genre_matrix, columns=genre_cols, index=bnb_df.index)
    bnb_features = pd.concat(
        [
            bnb_df[
                [
                    "high_vote_average",
                    "high_runtime",
                    "high_vote_count",
                    "high_budget",
                    "high_revenue",
                    "summer_release",
                    "recent_release",
                ]
            ],
            genre_binary_df,
        ],
        axis=1,
    )
    X_train_bnb = bnb_features.loc[train_idx].to_numpy()
    X_test_bnb = bnb_features.loc[test_idx].to_numpy()

    models = {
        "Multinomial NB": (MultinomialNB(alpha=1.0), X_train_mnb, X_test_mnb, mnb_feature_names),
        "Gaussian NB": (GaussianNB(), X_train_gnb, X_test_gnb, list(gnb_features.columns)),
        "Bernoulli NB": (BernoulliNB(alpha=1.0), X_train_bnb, X_test_bnb, list(bnb_features.columns)),
    }

    results = []
    confusion_paths = []
    preview_paths = []

    preview_map = {
        "Multinomial NB": pd.DataFrame(X_train_mnb, columns=mnb_feature_names).head(8).round(3),
        "Gaussian NB": pd.DataFrame(X_train_gnb, columns=gnb_features.columns).head(8).round(3),
        "Bernoulli NB": pd.DataFrame(X_train_bnb, columns=bnb_features.columns).head(8),
    }

    for model_name, preview_df in preview_map.items():
        preview_path = OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_preview.png"
        plot_dataframe_table(preview_df.iloc[:, : min(10, preview_df.shape[1])], f"{model_name} Training Data Preview", preview_path)
        preview_paths.append(preview_path.name)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    for idx, (model_name, (model, X_train, X_test, feature_names)) in enumerate(models.items()):
        model.fit(X_train, y.loc[train_idx])
        preds = model.predict(X_test)
        cm = confusion_matrix(y.loc[test_idx], preds)
        acc = accuracy_score(y.loc[test_idx], preds)
        plot_confusion(axes[idx], cm, model_name)
        results.append(
            {
                "model": model_name,
                "accuracy": round(float(acc), 4),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )

    finish_figure(OUTPUT_DIR / "nb_confusion_matrices.png")

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    save_csv(results_df, "nb_model_metrics.csv")
    plot_nb_accuracy(results_df.sort_values("model"), OUTPUT_DIR / "nb_accuracy_comparison.png")

    return {
        "results_df": results_df,
        "preview_paths": preview_paths,
        "best_accuracy": float(results_df.iloc[0]["accuracy"]),
    }


def run_decision_trees(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    y = df["label_popular_top25"].astype(int)
    full_df, genre_cols = build_multilabel_frame(df)
    feature_cols = ["vote_average", "vote_count", "runtime", "budget", "revenue", "release_year", "release_month"] + genre_cols
    X = full_df[feature_cols]

    tree_specs = [
        {"name": "Tree A", "criterion": "gini", "max_depth": 4, "min_samples_leaf": 5},
        {"name": "Tree B", "criterion": "entropy", "max_depth": 4, "min_samples_leaf": 5},
        {"name": "Tree C", "criterion": "gini", "max_depth": 5, "min_samples_leaf": 8},
    ]

    excluded_roots = set()
    tree_rows = []
    tree_accuracies = []
    best_model = None
    best_features = None
    best_accuracy = -1.0
    best_cm = None

    for idx, spec in enumerate(tree_specs, start=1):
        current_features = [feature for feature in feature_cols if feature not in excluded_roots]
        clf = DecisionTreeClassifier(
            criterion=spec["criterion"],
            max_depth=spec["max_depth"],
            min_samples_leaf=spec["min_samples_leaf"],
            random_state=42 + idx,
        )
        clf.fit(X.loc[train_idx, current_features], y.loc[train_idx])
        preds = clf.predict(X.loc[test_idx, current_features])
        cm = confusion_matrix(y.loc[test_idx], preds)
        acc = accuracy_score(y.loc[test_idx], preds)
        root_feature = current_features[clf.tree_.feature[0]]
        excluded_roots.add(root_feature)

        tree_rows.append(
            {
                "tree": spec["name"],
                "criterion": spec["criterion"],
                "max_depth": spec["max_depth"],
                "min_samples_leaf": spec["min_samples_leaf"],
                "root_feature": root_feature,
                "accuracy": round(float(acc), 4),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }
        )
        tree_accuracies.append({"model": spec["name"], "accuracy": round(float(acc), 4)})

        plt.figure(figsize=(20, 9))
        plot_tree(
            clf,
            feature_names=current_features,
            class_names=["Not Top25", "Top25"],
            filled=True,
            rounded=True,
            fontsize=8,
        )
        finish_figure(OUTPUT_DIR / f"decision_tree_{idx}.png")

        if acc > best_accuracy:
            best_accuracy = float(acc)
            best_model = clf
            best_features = current_features
            best_cm = cm

    save_csv(pd.DataFrame(tree_rows), "dt_model_metrics.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_confusion(ax, best_cm, "Best Decision Tree")
    finish_figure(OUTPUT_DIR / "dt_confusion_matrix.png")

    importances = pd.Series(best_model.feature_importances_, index=best_features)
    importances = importances[importances > 0].sort_values(ascending=False).head(10)
    plt.figure(figsize=(8.5, 4.8))
    plt.barh(importances.index[::-1], importances.values[::-1], color="#7aa2ff")
    plt.title("Best Decision Tree Feature Importances")
    plt.xlabel("Importance")
    finish_figure(OUTPUT_DIR / "dt_feature_importance.png")

    return {
        "best_accuracy": round(best_accuracy, 4),
        "best_cm": best_cm.tolist(),
        "tree_metrics": tree_rows,
    }


def run_regression_compare(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    y = df["label_popular_top25"].astype(int)
    full_df, genre_cols = build_multilabel_frame(df)
    feature_cols = ["vote_average", "vote_count", "runtime", "budget", "revenue", "release_year", "release_month"] + genre_cols
    X = full_df[feature_cols].copy()
    for col in ["vote_count", "budget", "revenue"]:
        X[col] = np.log1p(X[col])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.loc[train_idx])
    X_test = scaler.transform(X.loc[test_idx])

    logistic = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
    logistic.fit(X_train, y.loc[train_idx])
    logistic_preds = logistic.predict(X_test)
    logistic_cm = confusion_matrix(y.loc[test_idx], logistic_preds)
    logistic_acc = accuracy_score(y.loc[test_idx], logistic_preds)

    vectorizer = CountVectorizer(token_pattern=r"[^|]+")
    X_train_text = vectorizer.fit_transform(df.loc[train_idx, "genres"])
    X_test_text = vectorizer.transform(df.loc[test_idx, "genres"])
    numeric_cols = ["vote_average", "runtime", "vote_count", "budget", "revenue", "release_year", "release_month"]
    X_train_nb_num = df.loc[train_idx, numeric_cols].copy()
    X_test_nb_num = df.loc[test_idx, numeric_cols].copy()
    for col in ["vote_count", "budget", "revenue"]:
        X_train_nb_num[col] = np.log1p(X_train_nb_num[col])
        X_test_nb_num[col] = np.log1p(X_test_nb_num[col])
    X_train_mnb = np.hstack([X_train_nb_num.to_numpy(), X_train_text.toarray()])
    X_test_mnb = np.hstack([X_test_nb_num.to_numpy(), X_test_text.toarray()])
    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_mnb, y.loc[train_idx])
    mnb_preds = mnb.predict(X_test_mnb)
    mnb_cm = confusion_matrix(y.loc[test_idx], mnb_preds)
    mnb_acc = accuracy_score(y.loc[test_idx], mnb_preds)

    best_tree_features = [feature for feature in feature_cols if feature not in {"release_year", "release_month"}]
    tree = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=8, random_state=45)
    tree.fit(full_df.loc[train_idx, best_tree_features], y.loc[train_idx])
    tree_preds = tree.predict(full_df.loc[test_idx, best_tree_features])
    tree_cm = confusion_matrix(y.loc[test_idx], tree_preds)
    tree_acc = accuracy_score(y.loc[test_idx], tree_preds)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
    plot_confusion(axes[0], logistic_cm, "Logistic Regression")
    plot_confusion(axes[1], mnb_cm, "Multinomial NB")
    plot_confusion(axes[2], tree_cm, "Decision Tree")
    finish_figure(OUTPUT_DIR / "regression_model_confusion_matrices.png")

    compare_df = pd.DataFrame(
        [
            {"model": "Logistic Regression", "accuracy": round(float(logistic_acc), 4)},
            {"model": "Multinomial NB", "accuracy": round(float(mnb_acc), 4)},
            {"model": "Decision Tree", "accuracy": round(float(tree_acc), 4)},
        ]
    ).sort_values("accuracy", ascending=False)
    save_csv(compare_df, "regression_model_comparison.csv")
    plot_regression_compare(compare_df.sort_values("model"), OUTPUT_DIR / "regression_accuracy_comparison.png")

    coefs = pd.Series(logistic.coef_[0], index=feature_cols).sort_values(key=np.abs, ascending=False).head(10)
    coef_df = pd.DataFrame(
        {"feature": coefs.index, "coefficient": np.round(coefs.values, 4)}
    )
    save_csv(coef_df, "logistic_top_coefficients.csv")

    return {
        "comparison": compare_df.to_dict(orient="records"),
        "logistic_accuracy": round(float(logistic_acc), 4),
        "mnb_accuracy": round(float(mnb_acc), 4),
        "dt_accuracy": round(float(tree_acc), 4),
    }


def save_split_samples(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    train_cols = ["id", "title", "vote_average", "vote_count", "runtime", "budget", "revenue", "genres", "label_popular_top25"]
    test_cols = train_cols

    train_sample = df.loc[train_idx, train_cols].head(10).copy()
    test_sample = df.loc[test_idx, test_cols].head(10).copy()
    save_csv(train_sample, "train_sample.csv")
    save_csv(test_sample, "test_sample.csv")

    plot_dataframe_table(train_sample.head(8), "Training Set Preview", OUTPUT_DIR / "train_preview.png")
    plot_dataframe_table(test_sample.head(8), "Testing Set Preview", OUTPUT_DIR / "test_preview.png")


def main():
    ensure_dirs()
    df, train_idx, test_idx, split_meta = prepare_base_data()

    save_csv(df[["id", "title", "vote_average", "vote_count", "runtime", "budget", "revenue", "genres", "release_year", "release_month", "label_popular_top25"]], "module3_modeling_dataset.csv")
    save_split_samples(df, train_idx, test_idx)
    save_json(split_meta, "train_test_split_summary.json")
    plot_split_overview(df.loc[train_idx, "label_popular_top25"], df.loc[test_idx, "label_popular_top25"], OUTPUT_DIR / "train_test_split.png")

    nb_results = run_naive_bayes(df, train_idx, test_idx)
    dt_results = run_decision_trees(df, train_idx, test_idx)
    reg_results = run_regression_compare(df, train_idx, test_idx)

    plot_dt_overview_example(OUTPUT_DIR / "dt_overview_split_example.png")
    plot_dt_impurity_example(OUTPUT_DIR / "dt_impurity_example.png")

    summary = {
        "split": split_meta,
        "naive_bayes": nb_results["results_df"].to_dict(orient="records"),
        "decision_trees": dt_results["tree_metrics"],
        "regression_compare": reg_results["comparison"],
    }
    save_json(summary, "module3_summary.json")

    print("Module 3 outputs generated in outputs/module3 and copied into website/assets.")


if __name__ == "__main__":
    main()
