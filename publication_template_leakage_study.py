from pathlib import Path
import hashlib
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC


TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
SEEDS = [7, 13, 23, 31, 43]


def _clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _sha1(value):
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def labels(self):
        roots = [self.find(i) for i in range(len(self.parent))]
        remap = {root: idx for idx, root in enumerate(sorted(set(roots)))}
        return np.array([remap[root] for root in roots])


def _savefig(fig_dir, name):
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _load_data(data_path):
    df = pd.read_csv(data_path)
    for col in TEXT_COLS:
        df[col] = df[col].fillna("")
    df["combined_text"] = df[TEXT_COLS].agg(" ".join, axis=1)
    df["normalized_text"] = df["combined_text"].map(_clean_text)
    df["exact_signature"] = df["normalized_text"].map(_sha1)
    return df


def _near_duplicate_pairs(df, random_state=42, max_features=18000, n_neighbors=8):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    x_text = vectorizer.fit_transform(df["normalized_text"])
    n_components = min(120, max(2, x_text.shape[1] - 1))
    embedding = TruncatedSVD(n_components=n_components, random_state=random_state).fit_transform(x_text)
    embedding = normalize(embedding)

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="ball_tree")
    nn.fit(embedding)
    distances, indices = nn.kneighbors(embedding)

    rows = []
    seen = set()
    for i in range(indices.shape[0]):
        for pos in range(1, indices.shape[1]):
            j = int(indices[i, pos])
            if i == j:
                continue
            a, b = sorted((i, j))
            if (a, b) in seen:
                continue
            seen.add((a, b))
            cosine_similarity = 1 - (float(distances[i, pos]) ** 2 / 2)
            rows.append({"row_a": a, "row_b": b, "cosine_similarity": cosine_similarity})
    return pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)


def _near_duplicate_clusters(n_rows, pairs, threshold=0.98):
    uf = UnionFind(n_rows)
    for row in pairs[pairs["cosine_similarity"].ge(threshold)].itertuples(index=False):
        uf.union(int(row.row_a), int(row.row_b))
    return uf.labels()


def _cluster_summary(df, cluster_col):
    stats = (
        df.groupby(cluster_col)
        .agg(cluster_size=(cluster_col, "size"), fake_count=("fraudulent", "sum"))
        .reset_index()
    )
    clustered = stats[stats["cluster_size"] > 1].copy()
    mixed = clustered[
        (clustered["fake_count"] > 0) & (clustered["fake_count"] < clustered["cluster_size"])
    ]
    clustered_ids = set(clustered[cluster_col])
    return {
        "cluster_type": cluster_col,
        "cluster_count": int(len(clustered)),
        "rows_in_clusters": int(clustered["cluster_size"].sum()),
        "largest_cluster": int(stats["cluster_size"].max()),
        "mixed_label_clusters": int(len(mixed)),
        "fake_rate_clustered_rows": float(df[df[cluster_col].isin(clustered_ids)]["fraudulent"].mean()),
    }


def _choose_group_split(df, group_col, seed, test_size=0.25, candidates=30):
    y = df["fraudulent"].astype(int).to_numpy()
    groups = df[group_col].to_numpy()
    overall_rate = y.mean()
    splitter = GroupShuffleSplit(n_splits=candidates, test_size=test_size, random_state=seed)
    best = None
    best_score = np.inf
    for train_idx, test_idx in splitter.split(df, y, groups):
        if y[train_idx].sum() == 0 or y[test_idx].sum() == 0:
            continue
        score = abs(y[test_idx].mean() - overall_rate) + abs(len(test_idx) / len(df) - test_size)
        if score < best_score:
            best = (train_idx, test_idx)
            best_score = score
    return best


def _make_split(df, split_protocol, seed):
    idx = np.arange(len(df))
    if split_protocol == "random_stratified":
        return train_test_split(
            idx,
            test_size=0.25,
            stratify=df["fraudulent"],
            random_state=seed,
        )
    if split_protocol == "exact_duplicate_group":
        return _choose_group_split(df, "exact_signature", seed)
    if split_protocol == "near_template_group_098":
        return _choose_group_split(df, "near_template_cluster_098", seed)
    raise ValueError(f"Unknown split protocol: {split_protocol}")


def _make_model(model_name):
    tfidf = TfidfVectorizer(
        max_features=8000,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    if model_name == "Linear SVM":
        estimator = LinearSVC(class_weight="balanced", max_iter=4000, random_state=42)
    elif model_name == "Logistic Regression":
        estimator = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=42,
        )
    elif model_name == "Complement Naive Bayes":
        estimator = ComplementNB()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("tfidf", tfidf), ("model", estimator)])


def _score_model(model, x_test):
    estimator = model.named_steps["model"]
    if hasattr(estimator, "decision_function"):
        return model.decision_function(x_test)
    return model.predict_proba(x_test)[:, 1]


def _metrics(y_true, y_pred, y_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "fake_precision": precision_score(y_true, y_pred, zero_division=0),
        "fake_recall": recall_score(y_true, y_pred, zero_division=0),
        "fake_f1": f1_score(y_true, y_pred, zero_division=0),
        "predicted_fake_rate": float(np.mean(y_pred)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def _evaluate(df, model_name, split_protocol, seed):
    train_idx, test_idx = _make_split(df, split_protocol, seed)
    x_train = df.iloc[train_idx]["combined_text"]
    x_test = df.iloc[test_idx]["combined_text"]
    y_train = df.iloc[train_idx]["fraudulent"].astype(int)
    y_test = df.iloc[test_idx]["fraudulent"].astype(int)

    model = _make_model(model_name)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_score = _score_model(model, x_test)
    row = _metrics(y_test, y_pred, y_score)
    row.update(
        {
            "model": model_name,
            "split_protocol": split_protocol,
            "seed": seed,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "train_fake_rate": float(y_train.mean()),
            "test_fake_rate": float(y_test.mean()),
            "test_fake_count": int(y_test.sum()),
        }
    )
    return row


def _leakage_by_seed(df, seed, cluster_col):
    train_idx, test_idx = _make_split(df, "random_stratified", seed)
    train_groups = set(df.iloc[train_idx][cluster_col])
    test_groups = set(df.iloc[test_idx][cluster_col])
    overlap = train_groups.intersection(test_groups)
    leaked = df.iloc[test_idx][cluster_col].isin(overlap)
    return {
        "seed": seed,
        "cluster_type": cluster_col,
        "overlapping_clusters": int(len(overlap)),
        "test_rows_with_train_cluster": int(leaked.sum()),
        "share_test_rows_with_train_cluster": float(leaked.mean()),
    }


def _summarize_results(results):
    metrics = [
        "average_precision",
        "fake_precision",
        "fake_recall",
        "fake_f1",
        "false_positives",
        "false_negatives",
    ]
    summary = (
        results.groupby(["model", "split_protocol"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def _drop_summary(results):
    baseline = results[results["split_protocol"].eq("random_stratified")][
        ["model", "seed", "average_precision", "fake_recall", "fake_f1"]
    ].rename(
        columns={
            "average_precision": "random_average_precision",
            "fake_recall": "random_fake_recall",
            "fake_f1": "random_fake_f1",
        }
    )
    compared = results[~results["split_protocol"].eq("random_stratified")].merge(
        baseline, on=["model", "seed"], how="left"
    )
    compared["average_precision_drop_vs_random"] = (
        compared["random_average_precision"] - compared["average_precision"]
    )
    compared["fake_recall_drop_vs_random"] = compared["random_fake_recall"] - compared["fake_recall"]
    compared["fake_f1_drop_vs_random"] = compared["random_fake_f1"] - compared["fake_f1"]
    summary = (
        compared.groupby(["model", "split_protocol"])[
            [
                "average_precision_drop_vs_random",
                "fake_recall_drop_vs_random",
                "fake_f1_drop_vs_random",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    return summary


def _manual_review_candidates(df, pairs):
    candidates = pairs[
        pairs["cosine_similarity"].between(0.98, 0.995, inclusive="both")
    ].head(80).copy()
    for side in ["a", "b"]:
        row_col = f"row_{side}"
        candidates[f"job_id_{side}"] = df.iloc[candidates[row_col]]["job_id"].to_numpy()
        candidates[f"fraudulent_{side}"] = df.iloc[candidates[row_col]]["fraudulent"].to_numpy()
        candidates[f"title_{side}"] = df.iloc[candidates[row_col]]["title"].astype(str).str.slice(0, 120).to_numpy()
        candidates[f"description_{side}"] = (
            df.iloc[candidates[row_col]]["description"].astype(str).str.slice(0, 260).to_numpy()
        )
    return candidates


def _make_figures(results, drop_df, leakage_df, fig_dir):
    sns.set_theme(style="whitegrid", palette="Set2")
    metric_long = results.melt(
        id_vars=["model", "split_protocol", "seed"],
        value_vars=["average_precision", "fake_recall", "fake_f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(11, 6))
    sns.barplot(data=metric_long, y="model", x="score", hue="split_protocol")
    plt.title("Model Performance by Evaluation Protocol")
    plt.xlabel("Score")
    plt.ylabel("Model")
    plt.xlim(0, 1)
    _savefig(fig_dir, "model_performance_by_protocol")

    drop_plot = drop_df.copy()
    plt.figure(figsize=(10, 5.5))
    sns.barplot(
        data=drop_plot,
        y="model",
        x="fake_f1_drop_vs_random_mean",
        hue="split_protocol",
    )
    plt.title("Mean Fake F1 Drop Relative to Random Split")
    plt.xlabel("Mean fake F1 drop")
    plt.ylabel("Model")
    _savefig(fig_dir, "fake_f1_drop_vs_random")

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=leakage_df,
        x="cluster_type",
        y="share_test_rows_with_train_cluster",
        errorbar="sd",
    )
    plt.title("Random Split Leakage by Cluster Definition")
    plt.xlabel("Cluster definition")
    plt.ylabel("Share of test rows with train-cluster overlap")
    plt.ylim(0, 1)
    _savefig(fig_dir, "random_split_leakage_by_cluster_type")


def run_publication_template_leakage_study(
    data_path="Data/fake_jobs_cleaned.csv",
    output_dir="publication_outputs",
):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = _load_data(data_path)
    pairs = _near_duplicate_pairs(df, random_state=42)
    df["near_template_cluster_098"] = _near_duplicate_clusters(len(df), pairs, threshold=0.98)

    cluster_summary = pd.DataFrame(
        [
            _cluster_summary(df, "exact_signature"),
            _cluster_summary(df, "near_template_cluster_098"),
        ]
    )
    cluster_summary.to_csv(table_dir / "cluster_summary.csv", index=False)

    leakage_rows = []
    for seed in SEEDS:
        leakage_rows.append(_leakage_by_seed(df, seed, "exact_signature"))
        leakage_rows.append(_leakage_by_seed(df, seed, "near_template_cluster_098"))
    leakage_df = pd.DataFrame(leakage_rows)
    leakage_df.to_csv(table_dir / "random_split_leakage_by_seed.csv", index=False)
    leakage_summary = (
        leakage_df.groupby("cluster_type")
        .agg(
            overlapping_clusters_mean=("overlapping_clusters", "mean"),
            test_rows_with_train_cluster_mean=("test_rows_with_train_cluster", "mean"),
            share_test_rows_with_train_cluster_mean=("share_test_rows_with_train_cluster", "mean"),
            share_test_rows_with_train_cluster_std=("share_test_rows_with_train_cluster", "std"),
        )
        .reset_index()
    )
    leakage_summary.to_csv(table_dir / "random_split_leakage_summary.csv", index=False)

    manual_candidates = _manual_review_candidates(df, pairs)
    manual_candidates.to_csv(table_dir / "near_template_manual_review_candidates.csv", index=False)
    pairs.head(250).to_csv(table_dir / "near_template_top_pairs.csv", index=False)

    rows = []
    models = ["Linear SVM", "Logistic Regression", "Complement Naive Bayes"]
    splits = ["random_stratified", "exact_duplicate_group", "near_template_group_098"]
    for seed in SEEDS:
        for split_protocol in splits:
            for model_name in models:
                rows.append(_evaluate(df, model_name, split_protocol, seed))
                pd.DataFrame(rows).to_csv(
                    table_dir / "repeated_split_model_results_partial.csv", index=False
                )
                print(f"Finished seed={seed} split={split_protocol} model={model_name}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(table_dir / "repeated_split_model_results.csv", index=False)

    summary = _summarize_results(results)
    summary.to_csv(table_dir / "repeated_split_model_summary.csv", index=False)

    drop_df = _drop_summary(results)
    drop_df.to_csv(table_dir / "split_performance_drop_summary.csv", index=False)

    _make_figures(results, drop_df, leakage_df, fig_dir)
    return {
        "cluster_summary": cluster_summary,
        "leakage_summary": leakage_summary,
        "results": results,
        "summary": summary,
        "drop_summary": drop_df,
    }


if __name__ == "__main__":
    run_publication_template_leakage_study()
