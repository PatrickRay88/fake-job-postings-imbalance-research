from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.exceptions import ConvergenceWarning

from artifact_robustness_audit import _evaluate_split


TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]


def _display_and_save(df, table_dir, name, display_fn=None, index=False):
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / f"{name}.csv"
    df.to_csv(path, index=index)
    if display_fn is not None:
        display_fn(df)
    print(f"Saved: {path}")
    return path


def _savefig(fig_dir, name):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    return path


def _clean(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


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


def _prepare_data(data_path):
    df = pd.read_csv(data_path)
    for col in TEXT_COLS:
        df[col] = df[col].fillna("")
    df["combined_text"] = df[TEXT_COLS].agg(" ".join, axis=1)
    df["normalized_combined_text"] = df["combined_text"].map(_clean)
    return df


def _near_duplicate_pairs(df, max_features=18000, n_neighbors=8, random_state=42):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    x_text = vectorizer.fit_transform(df["normalized_combined_text"])
    n_components = min(120, max(2, x_text.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    x_embedding = normalize(svd.fit_transform(x_text))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", algorithm="ball_tree")
    nn.fit(x_embedding)
    distances, indices = nn.kneighbors(x_embedding)
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


def _cluster_from_pairs(n_rows, pairs, threshold):
    uf = UnionFind(n_rows)
    for row in pairs[pairs["cosine_similarity"].ge(threshold)].itertuples(index=False):
        uf.union(int(row.row_a), int(row.row_b))
    return uf.labels()


def _choose_group_split(df, group_col, random_state):
    y = df["fraudulent"].astype(int).to_numpy()
    groups = df[group_col].to_numpy()
    overall_rate = y.mean()
    gss = GroupShuffleSplit(n_splits=75, test_size=0.25, random_state=random_state)
    best = None
    best_score = np.inf
    for train_idx, test_idx in gss.split(df, y, groups):
        if y[train_idx].sum() == 0 or y[test_idx].sum() == 0:
            continue
        score = abs(y[test_idx].mean() - overall_rate) + abs(len(test_idx) / len(df) - 0.25)
        if score < best_score:
            best = (train_idx, test_idx)
            best_score = score
    return best


def _run_near_duplicate_audit(df, table_dir, fig_dir, display_fn, random_state):
    pairs = _near_duplicate_pairs(df, random_state=random_state)
    top_pairs = pairs.head(100).copy()
    for side in ["a", "b"]:
        idx_col = f"row_{side}"
        top_pairs[f"job_id_{side}"] = df.iloc[top_pairs[idx_col]]["job_id"].to_numpy()
        top_pairs[f"fraudulent_{side}"] = df.iloc[top_pairs[idx_col]]["fraudulent"].to_numpy()
        top_pairs[f"title_{side}"] = (
            df.iloc[top_pairs[idx_col]]["title"].astype(str).str.slice(0, 90).to_numpy()
        )
    _display_and_save(top_pairs.round(4), table_dir, "near_duplicate_top_pairs", display_fn)

    idx = np.arange(len(df))
    random_train, random_test = train_test_split(
        idx, test_size=0.25, stratify=df["fraudulent"], random_state=random_state
    )
    summary_rows = []
    leakage_rows = []

    for threshold in [0.95, 0.98, 0.99]:
        cluster_col = f"near_duplicate_cluster_{int(threshold * 100)}"
        df[cluster_col] = _cluster_from_pairs(len(df), pairs, threshold)
        cluster_stats = df.groupby(cluster_col).agg(
            cluster_size=(cluster_col, "size"),
            fake_count=("fraudulent", "sum"),
        )
        clustered = cluster_stats[cluster_stats["cluster_size"] > 1]
        mixed = clustered[
            (clustered["fake_count"] > 0) & (clustered["fake_count"] < clustered["cluster_size"])
        ]
        clustered_ids = set(clustered.index)
        summary_rows.append(
            {
                "similarity_threshold": threshold,
                "near_duplicate_clusters": int(len(clustered)),
                "rows_in_near_duplicate_clusters": int(clustered["cluster_size"].sum()),
                "largest_cluster": int(cluster_stats["cluster_size"].max()),
                "mixed_label_clusters": int(len(mixed)),
                "fake_rate_clustered_rows": float(
                    df[df[cluster_col].isin(clustered_ids)]["fraudulent"].mean()
                ),
            }
        )

        train_groups = set(df.iloc[random_train][cluster_col])
        test_groups = set(df.iloc[random_test][cluster_col])
        overlap = train_groups.intersection(test_groups)
        test_leaked = df.iloc[random_test][cluster_col].isin(overlap)
        leakage_rows.append(
            {
                "similarity_threshold": threshold,
                "overlapping_train_test_clusters": int(len(overlap)),
                "test_rows_with_train_near_duplicate_cluster": int(test_leaked.sum()),
                "share_test_rows_with_train_near_duplicate_cluster": float(test_leaked.mean()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    leakage = pd.DataFrame(leakage_rows)
    _display_and_save(summary.round(4), table_dir, "near_duplicate_cluster_summary", display_fn)
    _display_and_save(leakage.round(4), table_dir, "near_duplicate_random_split_leakage", display_fn)

    group_train, group_test = _choose_group_split(df, "near_duplicate_cluster_98", random_state)
    random_metrics, _, _, _, _ = _evaluate_split(
        "random_stratified_split", df, random_train, random_test, random_state
    )
    near_metrics, _, _, _, _ = _evaluate_split(
        "near_duplicate_group_split_98", df, group_train, group_test, random_state
    )
    split_df = pd.DataFrame([random_metrics, near_metrics])
    cols = [
        "split_name",
        "train_rows",
        "test_rows",
        "test_fake_rate",
        "average_precision",
        "fake_precision",
        "fake_recall",
        "fake_f1",
        "false_positives",
        "false_negatives",
    ]
    _display_and_save(split_df[cols].round(4), table_dir, "near_duplicate_split_comparison", display_fn)

    plot_long = split_df.melt(
        id_vars="split_name",
        value_vars=["average_precision", "fake_precision", "fake_recall", "fake_f1"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=plot_long, y="split_name", x="score", hue="metric")
    plt.title("Near-Duplicate Group Split Comparison")
    plt.xlabel("Score")
    plt.ylabel("Split strategy")
    plt.xlim(0, 1)
    _savefig(fig_dir, "near_duplicate_split_comparison")

    plt.figure(figsize=(7.5, 4.5))
    sns.barplot(
        data=summary,
        x="similarity_threshold",
        y="rows_in_near_duplicate_clusters",
        color="#4c78a8",
    )
    plt.title("Rows in Near-Duplicate Clusters")
    plt.xlabel("Cosine similarity threshold")
    plt.ylabel("Rows")
    _savefig(fig_dir, "near_duplicate_cluster_rows")
    return summary, leakage, split_df


def _run_fake_archetype_analysis(df, table_dir, fig_dir, display_fn, random_state):
    fake_df = df[df["fraudulent"].eq(1)].copy()
    vectorizer = TfidfVectorizer(
        max_features=9000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    x_fake = vectorizer.fit_transform(fake_df["combined_text"])
    n_topics = 6
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=random_state, max_iter=600)
    weights = nmf.fit_transform(x_fake)
    fake_df["fake_archetype"] = weights.argmax(axis=1)
    terms = np.array(vectorizer.get_feature_names_out())

    term_rows = []
    for topic_idx, topic_weights in enumerate(nmf.components_):
        top_idx = topic_weights.argsort()[::-1][:18]
        term_rows.append(
            {
                "fake_archetype": topic_idx,
                "top_terms": ", ".join(terms[top_idx]),
                "fake_posting_count": int((fake_df["fake_archetype"] == topic_idx).sum()),
            }
        )
    terms_df = pd.DataFrame(term_rows)
    _display_and_save(terms_df, table_dir, "fake_archetype_top_terms", display_fn)

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, stratify=df["fraudulent"], random_state=random_state
    )
    _, _, x_test, y_test, scores = _evaluate_split(
        "random_stratified_split", df, train_idx, test_idx, random_state
    )
    test_results = x_test[["job_id", "title", "combined_text"]].copy()
    test_results["actual"] = y_test.to_numpy()
    test_results["score_for_fake"] = scores
    test_results["predicted"] = (scores >= 0).astype(int)
    test_fake = test_results[test_results["actual"].eq(1)].copy()
    archetype_lookup = fake_df.set_index("job_id")["fake_archetype"]
    test_fake["fake_archetype"] = test_fake["job_id"].map(archetype_lookup)

    detection = (
        test_fake.groupby("fake_archetype")
        .agg(
            holdout_fake_count=("job_id", "size"),
            detected_count=("predicted", "sum"),
            mean_score_for_fake=("score_for_fake", "mean"),
        )
        .reset_index()
    )
    detection["fake_recall_within_archetype"] = (
        detection["detected_count"] / detection["holdout_fake_count"]
    )
    detection = detection.merge(terms_df, on="fake_archetype", how="left")
    _display_and_save(detection.round(4), table_dir, "fake_archetype_detection_summary", display_fn)

    examples = test_fake.sort_values("score_for_fake").groupby("fake_archetype").head(5).copy()
    examples["text_snippet"] = examples["combined_text"].astype(str).str.slice(0, 300)
    examples = examples[
        ["fake_archetype", "job_id", "title", "score_for_fake", "predicted", "text_snippet"]
    ]
    _display_and_save(examples.round(4), table_dir, "fake_archetype_low_score_examples", display_fn)

    plot_df = detection.sort_values("fake_recall_within_archetype").copy()
    plot_df["fake_archetype"] = plot_df["fake_archetype"].astype(str)
    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=plot_df, x="fake_recall_within_archetype", y="fake_archetype", color="#4c78a8")
    plt.title("Detection Recall by Fake-Posting Archetype")
    plt.xlabel("Fake recall within archetype")
    plt.ylabel("Fake archetype")
    plt.xlim(0, 1)
    _savefig(fig_dir, "fake_archetype_detection_recall")
    return terms_df, detection


def run_future_work_extensions(
    data_path="Data/fake_jobs_cleaned.csv",
    output_dir="future_work_outputs",
    display_fn=None,
    random_state=42,
):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    sns.set_theme(style="whitegrid", palette="Set2")
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_data(data_path)
    near_summary, near_leakage, near_split = _run_near_duplicate_audit(
        df, table_dir, fig_dir, display_fn, random_state
    )
    archetype_terms, archetype_detection = _run_fake_archetype_analysis(
        df, table_dir, fig_dir, display_fn, random_state
    )
    return {
        "near_duplicate_cluster_summary": near_summary,
        "near_duplicate_random_split_leakage": near_leakage,
        "near_duplicate_split_comparison": near_split,
        "fake_archetype_top_terms": archetype_terms,
        "fake_archetype_detection_summary": archetype_detection,
    }
