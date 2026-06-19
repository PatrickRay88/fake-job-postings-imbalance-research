from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLS = [
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
BINARY_COLS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "has_salary_range",
    "has_benefits",
    "has_company_profile",
    "has_department",
]
LENGTH_COLS = [
    "title_length",
    "description_length",
    "requirements_length",
    "company_profile_length",
    "benefits_length",
    "title_word_count",
    "description_word_count",
    "requirements_word_count",
    "company_profile_word_count",
    "benefits_word_count",
]
NUMERIC_COLS = BINARY_COLS + LENGTH_COLS


def _safe_display(display_fn, df):
    if display_fn is not None:
        display_fn(df)


def _savefig(fig_dir, name):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()
    return path


def _display_and_save(df, table_dir, name, display_fn=None, index=False):
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / f"{name}.csv"
    df.to_csv(path, index=index)
    _safe_display(display_fn, df)
    print(f"Saved: {path}")
    return path


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _make_preprocessor(feature_group="all", max_text_features=10000):
    transformers = []

    if feature_group in ["all", "text_only", "text_plus_credibility"]:
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    max_features=max_text_features,
                    min_df=2,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
                "combined_text",
            )
        )

    if feature_group in ["all", "metadata_only"]:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                CATEGORICAL_COLS,
            )
        )

    if feature_group in ["all", "metadata_only"]:
        numeric_cols = NUMERIC_COLS
    elif feature_group == "credibility_binary":
        numeric_cols = BINARY_COLS
    elif feature_group == "length_numeric":
        numeric_cols = LENGTH_COLS
    elif feature_group == "text_plus_credibility":
        numeric_cols = BINARY_COLS
    else:
        numeric_cols = []

    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_linear_svm_pipeline(
    class_weight="balanced", feature_group="all", max_text_features=10000, random_state=42
):
    return Pipeline(
        [
            (
                "features",
                _make_preprocessor(
                    feature_group=feature_group, max_text_features=max_text_features
                ),
            ),
            (
                "model",
                LinearSVC(
                    class_weight=class_weight,
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _get_model_scores(model, x_values):
    if hasattr(model, "decision_function"):
        return model.decision_function(x_values)
    return model.predict_proba(x_values)[:, 1]


def _classification_metrics_from_scores(y_true, scores, threshold=0.0):
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "fake_precision": precision_score(y_true, y_pred, zero_division=0),
        "fake_recall": recall_score(y_true, y_pred, zero_division=0),
        "fake_f1": f1_score(y_true, y_pred, zero_division=0),
        "predicted_fake_rate": y_pred.mean(),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def _evaluate_pipeline(name, model, x_eval, y_eval, threshold=0.0):
    scores = _get_model_scores(model, x_eval)
    row = _classification_metrics_from_scores(y_eval, scores, threshold=threshold)
    row["model"] = name
    return row, scores


def _prepare_data(data_path, random_state):
    cleaned_jobs = pd.read_csv(data_path)
    for col in TEXT_COLS + CATEGORICAL_COLS:
        cleaned_jobs[col] = cleaned_jobs[col].fillna("")
    cleaned_jobs["combined_text"] = cleaned_jobs[TEXT_COLS].agg(" ".join, axis=1)

    x_values = cleaned_jobs.drop(columns=["fraudulent"])
    y_values = cleaned_jobs["fraudulent"].astype(int)
    return train_test_split(
        x_values,
        y_values,
        test_size=0.25,
        stratify=y_values,
        random_state=random_state,
    )


def _run_prevalence_stress_test(
    y_true,
    scores,
    table_dir,
    fig_dir,
    display_fn,
    random_state,
    prevalence_values=None,
    sample_size=3000,
    repeats=200,
    threshold=0.0,
):
    if prevalence_values is None:
        prevalence_values = [0.01, 0.025, 0.05, 0.10, 0.20]

    rng = np.random.default_rng(random_state)
    y_array = np.asarray(y_true)
    score_array = np.asarray(scores)
    pos_idx = np.where(y_array == 1)[0]
    neg_idx = np.where(y_array == 0)[0]
    rows = []

    for prevalence in prevalence_values:
        n_pos = max(1, int(round(sample_size * prevalence)))
        n_neg = sample_size - n_pos
        for repeat in range(repeats):
            sampled_pos = rng.choice(pos_idx, size=n_pos, replace=n_pos > len(pos_idx))
            sampled_neg = rng.choice(neg_idx, size=n_neg, replace=n_neg > len(neg_idx))
            sampled_idx = np.concatenate([sampled_pos, sampled_neg])
            rng.shuffle(sampled_idx)
            metrics = _classification_metrics_from_scores(
                y_array[sampled_idx], score_array[sampled_idx], threshold=threshold
            )
            metrics["target_fake_prevalence"] = prevalence
            metrics["repeat"] = repeat
            rows.append(metrics)

    prevalence_results = pd.DataFrame(rows)
    prevalence_summary = (
        prevalence_results.groupby("target_fake_prevalence")
        .agg(
            fake_precision_mean=("fake_precision", "mean"),
            fake_precision_std=("fake_precision", "std"),
            fake_recall_mean=("fake_recall", "mean"),
            fake_recall_std=("fake_recall", "std"),
            fake_f1_mean=("fake_f1", "mean"),
            fake_f1_std=("fake_f1", "std"),
            predicted_fake_rate_mean=("predicted_fake_rate", "mean"),
            false_positives_mean=("false_positives", "mean"),
            false_negatives_mean=("false_negatives", "mean"),
        )
        .reset_index()
    )

    _display_and_save(
        prevalence_summary.round(4),
        table_dir,
        "prevalence_stress_test_summary",
        display_fn,
    )
    prevalence_results.to_csv(table_dir / "prevalence_stress_test_repeats.csv", index=False)

    prevalence_long = prevalence_summary.melt(
        id_vars="target_fake_prevalence",
        value_vars=["fake_precision_mean", "fake_recall_mean", "fake_f1_mean"],
        var_name="metric",
        value_name="score",
    )
    prevalence_long["metric"] = prevalence_long["metric"].map(
        {
            "fake_precision_mean": "Fake precision",
            "fake_recall_mean": "Fake recall",
            "fake_f1_mean": "Fake F1",
        }
    )

    plt.figure(figsize=(8.5, 5))
    sns.lineplot(
        data=prevalence_long,
        x="target_fake_prevalence",
        y="score",
        hue="metric",
        marker="o",
    )
    plt.title("Prevalence Stress Test: Metrics as Fake-Job Rate Changes")
    plt.xlabel("Simulated fake-job prevalence")
    plt.ylabel("Mean score across resamples")
    plt.ylim(0, 1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    _savefig(fig_dir, "prevalence_stress_precision_recall_f1")

    plt.figure(figsize=(8.5, 5))
    sns.lineplot(
        data=prevalence_summary,
        x="target_fake_prevalence",
        y="predicted_fake_rate_mean",
        marker="o",
        color="#4c78a8",
    )
    plt.title("Predicted Fake Rate Under Different Evaluation Prevalence Levels")
    plt.xlabel("Simulated fake-job prevalence")
    plt.ylabel("Mean share predicted fake")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    _savefig(fig_dir, "prevalence_stress_predicted_fake_rate")
    return prevalence_summary


def _run_review_budget_analysis(y_true, scores, table_dir, fig_dir, display_fn):
    budgets = [0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20]
    y_array = np.asarray(y_true)
    score_array = np.asarray(scores)
    order = np.argsort(score_array)[::-1]
    total_fake = int(y_array.sum())
    rows = []

    for budget in budgets:
        reviewed_count = max(1, int(round(len(y_array) * budget)))
        reviewed_idx = order[:reviewed_count]
        reviewed_y = y_array[reviewed_idx]
        true_positives = int(reviewed_y.sum())
        false_positives = int(reviewed_count - true_positives)
        rows.append(
            {
                "review_budget_rate": budget,
                "reviewed_count": reviewed_count,
                "true_positives_caught": true_positives,
                "false_positives_reviewed": false_positives,
                "false_negatives_left": int(total_fake - true_positives),
                "precision_among_reviewed": true_positives / reviewed_count,
                "fake_recall_at_budget": true_positives / total_fake if total_fake else 0,
                "share_of_all_fake_caught": true_positives / total_fake
                if total_fake
                else 0,
            }
        )

    review_budget_results = pd.DataFrame(rows)
    _display_and_save(
        review_budget_results.round(4), table_dir, "review_budget_analysis", display_fn
    )

    review_metric_long = review_budget_results.melt(
        id_vars="review_budget_rate",
        value_vars=["precision_among_reviewed", "fake_recall_at_budget"],
        var_name="metric",
        value_name="score",
    )
    review_metric_long["metric"] = review_metric_long["metric"].map(
        {
            "precision_among_reviewed": "Precision among reviewed",
            "fake_recall_at_budget": "Fake recall at budget",
        }
    )

    plt.figure(figsize=(8.5, 5))
    sns.lineplot(
        data=review_metric_long,
        x="review_budget_rate",
        y="score",
        hue="metric",
        marker="o",
    )
    plt.title("Review Budget Tradeoff")
    plt.xlabel("Share of postings sent to review")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    _savefig(fig_dir, "review_budget_precision_recall")

    review_count_long = review_budget_results.melt(
        id_vars="review_budget_rate",
        value_vars=[
            "true_positives_caught",
            "false_positives_reviewed",
            "false_negatives_left",
        ],
        var_name="outcome",
        value_name="count",
    )
    plt.figure(figsize=(8.5, 5))
    sns.lineplot(
        data=review_count_long,
        x="review_budget_rate",
        y="count",
        hue="outcome",
        marker="o",
    )
    plt.title("Review Budget Outcomes")
    plt.xlabel("Share of postings sent to review")
    plt.ylabel("Number of holdout postings")
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    _savefig(fig_dir, "review_budget_outcomes")
    return review_budget_results


def _run_expanded_cost_sensitivity(y_true, scores, table_dir, fig_dir, display_fn):
    score_quantiles = np.linspace(0, 1, 501)
    threshold_grid = np.unique(np.quantile(scores, score_quantiles))
    threshold_grid = np.unique(np.concatenate([threshold_grid, [0.0]]))
    threshold_metric_grid = pd.DataFrame(
        [_classification_metrics_from_scores(y_true, scores, t) for t in threshold_grid]
    )
    threshold_metric_grid.to_csv(table_dir / "expanded_threshold_metric_grid.csv", index=False)

    expanded_cost_rows = []
    expanded_cost_grid_rows = []
    for fn_cost in [1, 2, 5, 10, 20, 50, 100]:
        temp = threshold_metric_grid.copy()
        temp["false_positive_cost_value"] = 1
        temp["false_negative_cost_value"] = fn_cost
        temp["false_positive_cost"] = temp["false_positives"]
        temp["false_negative_cost"] = temp["false_negatives"] * fn_cost
        temp["total_cost"] = temp["false_positive_cost"] + temp["false_negative_cost"]
        expanded_cost_grid_rows.append(temp)
        best = temp.sort_values(["total_cost", "false_negatives", "false_positives"]).iloc[0]
        expanded_cost_rows.append(
            {
                "false_positive_cost": 1,
                "false_negative_cost": fn_cost,
                "selected_threshold": best["threshold"],
                "selected_flagged_rate": best["predicted_fake_rate"],
                "false_positives": int(best["false_positives"]),
                "false_negatives": int(best["false_negatives"]),
                "fake_precision": best["fake_precision"],
                "fake_recall": best["fake_recall"],
                "fake_f1": best["fake_f1"],
                "total_cost": best["total_cost"],
            }
        )

    expanded_cost_sensitivity = pd.DataFrame(expanded_cost_rows)
    expanded_cost_grid = pd.concat(expanded_cost_grid_rows, ignore_index=True)
    _display_and_save(
        expanded_cost_sensitivity.round(4),
        table_dir,
        "expanded_cost_sensitivity_threshold_selection",
        display_fn,
    )
    expanded_cost_grid.to_csv(table_dir / "expanded_cost_sensitivity_grid.csv", index=False)

    cost_plot_long = expanded_cost_sensitivity.melt(
        id_vars="false_negative_cost",
        value_vars=["selected_flagged_rate", "fake_precision", "fake_recall", "fake_f1"],
        var_name="metric",
        value_name="value",
    )
    cost_plot_long["metric"] = cost_plot_long["metric"].map(
        {
            "selected_flagged_rate": "Selected flagged rate",
            "fake_precision": "Fake precision",
            "fake_recall": "Fake recall",
            "fake_f1": "Fake F1",
        }
    )
    plt.figure(figsize=(9, 5.2))
    sns.lineplot(
        data=cost_plot_long,
        x="false_negative_cost",
        y="value",
        hue="metric",
        marker="o",
    )
    plt.title("Expanded Cost Sensitivity Tradeoffs")
    plt.xlabel("False negative cost, false positive cost fixed at 1")
    plt.ylabel("Value")
    plt.ylim(0, 1)
    _savefig(fig_dir, "expanded_cost_sensitivity_tradeoffs")

    cost_outcome_long = expanded_cost_sensitivity.melt(
        id_vars="false_negative_cost",
        value_vars=["false_positives", "false_negatives"],
        var_name="error_type",
        value_name="count",
    )
    plt.figure(figsize=(8.5, 5))
    sns.lineplot(
        data=cost_outcome_long,
        x="false_negative_cost",
        y="count",
        hue="error_type",
        marker="o",
    )
    plt.title("Expanded Cost Sensitivity Error Counts")
    plt.xlabel("False negative cost, false positive cost fixed at 1")
    plt.ylabel("Holdout error count")
    _savefig(fig_dir, "expanded_cost_sensitivity_error_counts")
    return expanded_cost_sensitivity


def _resample_training_data(x_train, y_train, strategy, random_state):
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y_train.to_numpy() == 1)[0]
    neg_idx = np.where(y_train.to_numpy() == 0)[0]

    if strategy == "observed":
        selected_idx = np.arange(len(y_train))
    elif strategy == "undersample_1_to_1":
        sampled_neg = rng.choice(neg_idx, size=len(pos_idx), replace=False)
        selected_idx = np.concatenate([pos_idx, sampled_neg])
    elif strategy == "oversample_1_to_1":
        sampled_pos = rng.choice(pos_idx, size=len(neg_idx), replace=True)
        selected_idx = np.concatenate([neg_idx, sampled_pos])
    elif strategy == "oversample_20_percent_fake":
        target_pos = int(round(0.20 / 0.80 * len(neg_idx)))
        sampled_pos = rng.choice(
            pos_idx, size=target_pos, replace=target_pos > len(pos_idx)
        )
        selected_idx = np.concatenate([neg_idx, sampled_pos])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    rng.shuffle(selected_idx)
    return x_train.iloc[selected_idx].copy(), y_train.iloc[selected_idx].copy()


def _run_training_distribution_experiment(
    x_train, y_train, x_test, y_test, table_dir, fig_dir, display_fn, random_state
):
    training_variants = [
        ("Observed imbalance, unweighted", "observed", None),
        ("Observed imbalance, class_weight=balanced", "observed", "balanced"),
        ("Undersampled 1:1, unweighted", "undersample_1_to_1", None),
        ("Oversampled 1:1, unweighted", "oversample_1_to_1", None),
        ("Oversampled 20% fake, unweighted", "oversample_20_percent_fake", None),
    ]

    rows = []
    for label, sampling_strategy, class_weight in training_variants:
        x_resampled, y_resampled = _resample_training_data(
            x_train, y_train, sampling_strategy, random_state
        )
        model = _make_linear_svm_pipeline(
            class_weight=class_weight, feature_group="all", random_state=random_state
        )
        model.fit(x_resampled, y_resampled)
        metrics, _ = _evaluate_pipeline(label, model, x_test, y_test)
        metrics["training_rows"] = len(y_resampled)
        metrics["training_fake_rate"] = y_resampled.mean()
        metrics["sampling_strategy"] = sampling_strategy
        metrics["class_weight"] = "none" if class_weight is None else class_weight
        rows.append(metrics)

    results = pd.DataFrame(rows)
    display_cols = [
        "model",
        "sampling_strategy",
        "class_weight",
        "training_rows",
        "training_fake_rate",
        "accuracy",
        "balanced_accuracy",
        "average_precision",
        "fake_precision",
        "fake_recall",
        "fake_f1",
        "predicted_fake_rate",
        "false_positives",
        "false_negatives",
    ]
    _display_and_save(
        results[display_cols].round(4),
        table_dir,
        "training_distribution_experiment",
        display_fn,
    )

    metric_long = results.melt(
        id_vars="model",
        value_vars=["average_precision", "fake_precision", "fake_recall", "fake_f1"],
        var_name="metric",
        value_name="score",
    )
    metric_long["metric"] = metric_long["metric"].map(
        {
            "average_precision": "Average precision",
            "fake_precision": "Fake precision",
            "fake_recall": "Fake recall",
            "fake_f1": "Fake F1",
        }
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metric_long, y="model", x="score", hue="metric")
    plt.title("Training Distribution Experiment")
    plt.xlabel("Holdout score")
    plt.ylabel("Training strategy")
    plt.xlim(0, 1)
    _savefig(fig_dir, "training_distribution_experiment_metrics")
    return results


def _run_feature_group_ablation(
    x_train, y_train, x_test, y_test, table_dir, fig_dir, display_fn, random_state
):
    feature_group_specs = [
        ("All features", "all"),
        ("Text only", "text_only"),
        ("Metadata only", "metadata_only"),
        ("Credibility binary only", "credibility_binary"),
        ("Length numeric only", "length_numeric"),
        ("Text plus credibility binary", "text_plus_credibility"),
    ]

    rows = []
    for label, feature_group in feature_group_specs:
        model = _make_linear_svm_pipeline(
            class_weight="balanced", feature_group=feature_group, random_state=random_state
        )
        model.fit(x_train, y_train)
        metrics, _ = _evaluate_pipeline(label, model, x_test, y_test)
        metrics["feature_group"] = feature_group
        rows.append(metrics)

    results = pd.DataFrame(rows)
    display_cols = [
        "model",
        "feature_group",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "average_precision",
        "fake_precision",
        "fake_recall",
        "fake_f1",
        "predicted_fake_rate",
        "false_positives",
        "false_negatives",
    ]
    _display_and_save(results[display_cols].round(4), table_dir, "feature_group_ablation", display_fn)

    metric_long = results.melt(
        id_vars="model",
        value_vars=["average_precision", "fake_precision", "fake_recall", "fake_f1"],
        var_name="metric",
        value_name="score",
    )
    metric_long["metric"] = metric_long["metric"].map(
        {
            "average_precision": "Average precision",
            "fake_precision": "Fake precision",
            "fake_recall": "Fake recall",
            "fake_f1": "Fake F1",
        }
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metric_long, y="model", x="score", hue="metric")
    plt.title("Feature Group Ablation")
    plt.xlabel("Holdout score")
    plt.ylabel("Feature group")
    plt.xlim(0, 1)
    _savefig(fig_dir, "feature_group_ablation_metrics")
    return results


def _run_label_scarcity_experiment(
    x_train,
    y_train,
    x_test,
    y_test,
    table_dir,
    fig_dir,
    display_fn,
    random_state,
    repeats=3,
):
    rng = np.random.default_rng(random_state)
    pos_idx = np.where(y_train.to_numpy() == 1)[0]
    neg_idx = np.where(y_train.to_numpy() == 0)[0]
    rows = []

    for fake_count in [50, 100, 200, 400, "all"]:
        actual_count = len(pos_idx) if fake_count == "all" else int(fake_count)
        repeat_count = 1 if fake_count == "all" else repeats
        for repeat in range(repeat_count):
            sampled_pos = rng.choice(
                pos_idx, size=actual_count, replace=actual_count > len(pos_idx)
            )
            selected_idx = np.concatenate([neg_idx, sampled_pos])
            rng.shuffle(selected_idx)
            x_subset = x_train.iloc[selected_idx].copy()
            y_subset = y_train.iloc[selected_idx].copy()
            model = _make_linear_svm_pipeline(
                class_weight="balanced", feature_group="all", random_state=random_state
            )
            model.fit(x_subset, y_subset)
            metrics, _ = _evaluate_pipeline(
                f"{actual_count} fake labels", model, x_test, y_test
            )
            metrics["fake_labels_available"] = actual_count
            metrics["repeat"] = repeat
            metrics["training_rows"] = len(y_subset)
            metrics["training_fake_rate"] = y_subset.mean()
            rows.append(metrics)

    results = pd.DataFrame(rows)
    results.to_csv(table_dir / "label_scarcity_experiment_repeats.csv", index=False)
    summary = (
        results.groupby("fake_labels_available")
        .agg(
            repeats=("repeat", "count"),
            training_fake_rate_mean=("training_fake_rate", "mean"),
            average_precision_mean=("average_precision", "mean"),
            average_precision_std=("average_precision", "std"),
            fake_precision_mean=("fake_precision", "mean"),
            fake_precision_std=("fake_precision", "std"),
            fake_recall_mean=("fake_recall", "mean"),
            fake_recall_std=("fake_recall", "std"),
            fake_f1_mean=("fake_f1", "mean"),
            fake_f1_std=("fake_f1", "std"),
            false_positives_mean=("false_positives", "mean"),
            false_negatives_mean=("false_negatives", "mean"),
        )
        .reset_index()
        .sort_values("fake_labels_available")
    )
    _display_and_save(
        summary.round(4), table_dir, "label_scarcity_experiment_summary", display_fn
    )

    metric_long = summary.melt(
        id_vars="fake_labels_available",
        value_vars=[
            "average_precision_mean",
            "fake_precision_mean",
            "fake_recall_mean",
            "fake_f1_mean",
        ],
        var_name="metric",
        value_name="score",
    )
    metric_long["metric"] = metric_long["metric"].map(
        {
            "average_precision_mean": "Average precision",
            "fake_precision_mean": "Fake precision",
            "fake_recall_mean": "Fake recall",
            "fake_f1_mean": "Fake F1",
        }
    )
    plt.figure(figsize=(8.5, 5))
    sns.lineplot(
        data=metric_long,
        x="fake_labels_available",
        y="score",
        hue="metric",
        marker="o",
    )
    plt.title("Label Scarcity Experiment")
    plt.xlabel("Fake labels available during training")
    plt.ylabel("Mean holdout score")
    plt.ylim(0, 1)
    _savefig(fig_dir, "label_scarcity_experiment_metrics")
    return summary


def _prediction_group(row):
    if row["actual"] == 1 and row["predicted"] == 1:
        return "true_positive_fake_detected"
    if row["actual"] == 0 and row["predicted"] == 1:
        return "false_positive_real_flagged"
    if row["actual"] == 1 and row["predicted"] == 0:
        return "false_negative_fake_missed"
    return "true_negative_real_kept"


def _run_holdout_error_profile(x_test, y_test, scores, table_dir, fig_dir, display_fn):
    holdout_errors = x_test.copy()
    holdout_errors["actual"] = y_test.to_numpy()
    holdout_errors["score_for_fake"] = scores
    holdout_errors["predicted"] = (scores >= 0).astype(int)
    holdout_errors["prediction_group"] = holdout_errors.apply(_prediction_group, axis=1)

    profile = (
        holdout_errors.groupby("prediction_group")
        .agg(
            count=("prediction_group", "size"),
            mean_score_for_fake=("score_for_fake", "mean"),
            company_logo_rate=("has_company_logo", "mean"),
            company_profile_rate=("has_company_profile", "mean"),
            questions_rate=("has_questions", "mean"),
            salary_range_rate=("has_salary_range", "mean"),
            benefits_rate=("has_benefits", "mean"),
            mean_description_length=("description_length", "mean"),
            mean_requirements_length=("requirements_length", "mean"),
            mean_company_profile_length=("company_profile_length", "mean"),
            mean_benefits_length=("benefits_length", "mean"),
        )
        .reset_index()
    )
    _display_and_save(profile.round(4), table_dir, "holdout_error_group_profile", display_fn)

    contrast_rows = []
    for feature in [
        "has_company_logo",
        "has_company_profile",
        "has_questions",
        "has_salary_range",
        "has_benefits",
    ]:
        fp_mean = holdout_errors.loc[
            holdout_errors["prediction_group"].eq("false_positive_real_flagged"),
            feature,
        ].mean()
        tn_mean = holdout_errors.loc[
            holdout_errors["prediction_group"].eq("true_negative_real_kept"), feature
        ].mean()
        fn_mean = holdout_errors.loc[
            holdout_errors["prediction_group"].eq("false_negative_fake_missed"),
            feature,
        ].mean()
        tp_mean = holdout_errors.loc[
            holdout_errors["prediction_group"].eq("true_positive_fake_detected"),
            feature,
        ].mean()
        contrast_rows.append(
            {
                "feature": feature,
                "false_positive_minus_true_negative": fp_mean - tn_mean,
                "false_negative_minus_true_positive": fn_mean - tp_mean,
                "false_positive_rate": fp_mean,
                "true_negative_rate": tn_mean,
                "false_negative_rate": fn_mean,
                "true_positive_rate": tp_mean,
            }
        )

    contrast = pd.DataFrame(contrast_rows)
    _display_and_save(
        contrast.round(4), table_dir, "holdout_error_binary_feature_contrasts", display_fn
    )

    profile_plot = profile.melt(
        id_vars="prediction_group",
        value_vars=[
            "company_logo_rate",
            "company_profile_rate",
            "questions_rate",
            "salary_range_rate",
            "benefits_rate",
        ],
        var_name="feature_rate",
        value_name="rate",
    )
    plt.figure(figsize=(10, 5.8))
    sns.barplot(data=profile_plot, y="prediction_group", x="rate", hue="feature_rate")
    plt.title("Holdout Error Profile: Binary Feature Rates by Prediction Group")
    plt.xlabel("Rate")
    plt.ylabel("Prediction group")
    plt.xlim(0, 1)
    _savefig(fig_dir, "holdout_error_binary_feature_profile")
    return profile


def run_comprehensive_imbalance_experiments(
    data_path="Data/fake_jobs_cleaned.csv",
    research_table_dir="imbalance_research_outputs/tables",
    research_fig_dir="imbalance_research_outputs/figures",
    display_fn=None,
    random_state=42,
):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    sns.set_theme(style="whitegrid", palette="Set2")

    table_dir = Path(research_table_dir)
    fig_dir = Path(research_fig_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = _prepare_data(data_path, random_state)
    print("Train rows:", len(x_train), "Test rows:", len(x_test))
    print("Train fake rate:", round(y_train.mean(), 4), "Test fake rate:", round(y_test.mean(), 4))

    selected_model = _make_linear_svm_pipeline(
        class_weight="balanced", feature_group="all", random_state=random_state
    )
    selected_model.fit(x_train, y_train)
    selected_metrics, selected_scores = _evaluate_pipeline(
        "Linear SVM balanced full features", selected_model, x_test, y_test
    )
    selected_display = pd.DataFrame([selected_metrics])[
        [
            "model",
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "average_precision",
            "fake_precision",
            "fake_recall",
            "fake_f1",
            "predicted_fake_rate",
            "true_positives",
            "false_positives",
            "false_negatives",
            "true_negatives",
        ]
    ]
    _display_and_save(
        selected_display.round(4), table_dir, "selected_model_holdout_metrics", display_fn
    )

    outputs = {
        "selected_model_holdout_metrics": selected_display,
        "prevalence_stress_test_summary": _run_prevalence_stress_test(
            y_test, selected_scores, table_dir, fig_dir, display_fn, random_state
        ),
        "review_budget_analysis": _run_review_budget_analysis(
            y_test, selected_scores, table_dir, fig_dir, display_fn
        ),
        "expanded_cost_sensitivity_threshold_selection": _run_expanded_cost_sensitivity(
            y_test, selected_scores, table_dir, fig_dir, display_fn
        ),
        "training_distribution_experiment": _run_training_distribution_experiment(
            x_train, y_train, x_test, y_test, table_dir, fig_dir, display_fn, random_state
        ),
        "feature_group_ablation": _run_feature_group_ablation(
            x_train, y_train, x_test, y_test, table_dir, fig_dir, display_fn, random_state
        ),
        "label_scarcity_experiment_summary": _run_label_scarcity_experiment(
            x_train, y_train, x_test, y_test, table_dir, fig_dir, display_fn, random_state
        ),
        "holdout_error_group_profile": _run_holdout_error_profile(
            x_test, y_test, selected_scores, table_dir, fig_dir, display_fn
        ),
    }
    return outputs
