from pathlib import Path
import hashlib
import re
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
from sklearn.model_selection import GroupShuffleSplit, train_test_split
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
    if display_fn is not None:
        display_fn(df)
    print(f"Saved: {path}")
    return path


def _make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _sha1_text(value):
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _prepare_frame(data_path):
    df = pd.read_csv(data_path)
    for col in TEXT_COLS + CATEGORICAL_COLS:
        df[col] = df[col].fillna("")
    df["combined_text"] = df[TEXT_COLS].agg(" ".join, axis=1)

    signature_text = (
        df[TEXT_COLS]
        .apply(lambda row: " ||| ".join(_clean_text(value) for value in row), axis=1)
    )
    df["content_signature"] = signature_text.map(_sha1_text)
    return df


def _make_preprocessor(feature_group="all", max_text_features=10000):
    transformers = []

    if feature_group in ["all", "text_only", "text_no_credibility"]:
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

    if feature_group in ["all", "metadata_only", "text_no_credibility"]:
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

    if feature_group == "all":
        numeric_cols = BINARY_COLS + LENGTH_COLS
    elif feature_group == "metadata_only":
        numeric_cols = BINARY_COLS + LENGTH_COLS
    elif feature_group == "shortcuts_only":
        numeric_cols = BINARY_COLS + LENGTH_COLS
    elif feature_group == "credibility_flags_only":
        numeric_cols = BINARY_COLS
    elif feature_group == "length_only":
        numeric_cols = LENGTH_COLS
    elif feature_group == "text_no_credibility":
        numeric_cols = LENGTH_COLS
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


def _make_model(feature_group="all", class_weight="balanced", random_state=42):
    return Pipeline(
        [
            ("features", _make_preprocessor(feature_group=feature_group)),
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


def _metrics_from_scores(y_true, scores, threshold=0.0):
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


def _metrics_for_group(y_true, scores, threshold=0.0):
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision_denom = tp + fp
    recall_denom = tp + fn
    real_denom = tn + fp
    return {
        "row_count": int(len(y_true)),
        "fake_count": int(y_true.sum()),
        "fake_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "predicted_fake_rate": float(y_pred.mean()) if len(y_pred) else np.nan,
        "fake_precision": tp / precision_denom if precision_denom else np.nan,
        "fake_recall": tp / recall_denom if recall_denom else np.nan,
        "fake_f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate_among_real": fp / real_denom if real_denom else np.nan,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "mean_score_for_fake": float(np.mean(scores)) if len(scores) else np.nan,
    }


def _split_by_indices(df, train_idx, test_idx):
    feature_cols = [col for col in df.columns if col != "fraudulent"]
    x_train = df.iloc[train_idx][feature_cols].copy()
    x_test = df.iloc[test_idx][feature_cols].copy()
    y_train = df.iloc[train_idx]["fraudulent"].astype(int).copy()
    y_test = df.iloc[test_idx]["fraudulent"].astype(int).copy()
    return x_train, x_test, y_train, y_test


def _evaluate_split(split_name, df, train_idx, test_idx, random_state):
    x_train, x_test, y_train, y_test = _split_by_indices(df, train_idx, test_idx)
    model = _make_model(feature_group="all", class_weight="balanced", random_state=random_state)
    model.fit(x_train, y_train)
    scores = model.decision_function(x_test)
    metrics = _metrics_from_scores(y_test, scores)
    metrics.update(
        {
            "split_name": split_name,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_fake_rate": float(y_train.mean()),
            "test_fake_rate": float(y_test.mean()),
            "test_fake_count": int(y_test.sum()),
        }
    )
    return metrics, model, x_test, y_test, scores


def _choose_group_split(df, random_state):
    y = df["fraudulent"].astype(int).to_numpy()
    groups = df["content_signature"].to_numpy()
    overall_rate = y.mean()
    gss = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=random_state)
    best = None
    best_score = np.inf
    for train_idx, test_idx in gss.split(df, y, groups):
        test_rate = y[test_idx].mean()
        test_size_gap = abs(len(test_idx) / len(df) - 0.25)
        fake_rate_gap = abs(test_rate - overall_rate)
        score = fake_rate_gap + test_size_gap
        if score < best_score and y[train_idx].sum() > 0 and y[test_idx].sum() > 0:
            best = (train_idx, test_idx)
            best_score = score
    return best


def _run_duplicate_and_split_audit(df, table_dir, fig_dir, display_fn, random_state):
    group_stats = (
        df.groupby("content_signature")
        .agg(
            group_size=("content_signature", "size"),
            fake_count=("fraudulent", "sum"),
            real_count=("fraudulent", lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    duplicate_groups = group_stats[group_stats["group_size"] > 1]
    mixed_label_groups = duplicate_groups[
        (duplicate_groups["fake_count"] > 0) & (duplicate_groups["real_count"] > 0)
    ]
    duplicate_summary = pd.DataFrame(
        [
            {
                "rows": len(df),
                "unique_content_signatures": df["content_signature"].nunique(),
                "duplicate_signature_groups": len(duplicate_groups),
                "rows_in_duplicate_groups": int(duplicate_groups["group_size"].sum()),
                "largest_duplicate_group": int(group_stats["group_size"].max()),
                "mixed_label_duplicate_groups": len(mixed_label_groups),
                "fake_rate_duplicate_rows": df[
                    df["content_signature"].isin(duplicate_groups["content_signature"])
                ]["fraudulent"].mean(),
            }
        ]
    )
    _display_and_save(
        duplicate_summary.round(4), table_dir, "duplicate_signature_summary", display_fn
    )

    idx = np.arange(len(df))
    random_train_idx, random_test_idx = train_test_split(
        idx,
        test_size=0.25,
        stratify=df["fraudulent"],
        random_state=random_state,
    )
    random_train_groups = set(df.iloc[random_train_idx]["content_signature"])
    random_test_groups = set(df.iloc[random_test_idx]["content_signature"])
    overlapping_groups = random_train_groups.intersection(random_test_groups)
    random_leakage_summary = pd.DataFrame(
        [
            {
                "split": "random_stratified",
                "overlapping_content_signatures": len(overlapping_groups),
                "test_rows_with_train_duplicate_signature": int(
                    df.iloc[random_test_idx]["content_signature"].isin(overlapping_groups).sum()
                ),
                "share_test_rows_with_train_duplicate_signature": float(
                    df.iloc[random_test_idx]["content_signature"].isin(overlapping_groups).mean()
                ),
            }
        ]
    )
    _display_and_save(
        random_leakage_summary.round(4),
        table_dir,
        "random_split_duplicate_leakage_summary",
        display_fn,
    )

    group_train_idx, group_test_idx = _choose_group_split(df, random_state)
    ordered = df.sort_values("job_id").index.to_numpy()
    cutoff = int(round(len(ordered) * 0.75))
    job_train_idx = ordered[:cutoff]
    job_test_idx = ordered[cutoff:]

    split_results = []
    split_artifacts = {}
    for split_name, train_idx, test_idx in [
        ("random_stratified_split", random_train_idx, random_test_idx),
        ("duplicate_group_split", group_train_idx, group_test_idx),
        ("job_id_order_split", job_train_idx, job_test_idx),
    ]:
        metrics, model, x_test, y_test, scores = _evaluate_split(
            split_name, df, train_idx, test_idx, random_state
        )
        split_results.append(metrics)
        split_artifacts[split_name] = {
            "model": model,
            "x_test": x_test,
            "y_test": y_test,
            "scores": scores,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }

    split_df = pd.DataFrame(split_results)
    split_cols = [
        "split_name",
        "train_rows",
        "test_rows",
        "train_fake_rate",
        "test_fake_rate",
        "test_fake_count",
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
    _display_and_save(split_df[split_cols].round(4), table_dir, "split_robustness_comparison", display_fn)

    plot_long = split_df.melt(
        id_vars="split_name",
        value_vars=["average_precision", "fake_precision", "fake_recall", "fake_f1"],
        var_name="metric",
        value_name="score",
    )
    plot_long["metric"] = plot_long["metric"].map(
        {
            "average_precision": "Average precision",
            "fake_precision": "Fake precision",
            "fake_recall": "Fake recall",
            "fake_f1": "Fake F1",
        }
    )
    plt.figure(figsize=(10, 5.8))
    sns.barplot(data=plot_long, y="split_name", x="score", hue="metric")
    plt.title("Split Robustness Comparison")
    plt.xlabel("Holdout score")
    plt.ylabel("Split strategy")
    plt.xlim(0, 1)
    _savefig(fig_dir, "split_robustness_comparison")
    return split_df, split_artifacts


def _run_shortcut_feature_audit(
    df, train_idx, test_idx, table_dir, fig_dir, display_fn, random_state
):
    x_train, x_test, y_train, y_test = _split_by_indices(df, train_idx, test_idx)
    feature_specs = [
        ("All features", "all"),
        ("Text only", "text_only"),
        ("Text + metadata + length, no credibility flags", "text_no_credibility"),
        ("Metadata + flags + lengths only", "metadata_only"),
        ("Shortcut features only: flags + lengths", "shortcuts_only"),
        ("Credibility flags only", "credibility_flags_only"),
        ("Length features only", "length_only"),
    ]

    rows = []
    for label, feature_group in feature_specs:
        model = _make_model(feature_group=feature_group, class_weight="balanced", random_state=random_state)
        model.fit(x_train, y_train)
        scores = model.decision_function(x_test)
        metrics = _metrics_from_scores(y_test, scores)
        metrics["model"] = label
        metrics["feature_group"] = feature_group
        rows.append(metrics)

    shortcut_df = pd.DataFrame(rows)
    cols = [
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
    _display_and_save(shortcut_df[cols].round(4), table_dir, "shortcut_feature_audit", display_fn)

    plot_long = shortcut_df.melt(
        id_vars="model",
        value_vars=["average_precision", "fake_precision", "fake_recall", "fake_f1"],
        var_name="metric",
        value_name="score",
    )
    plot_long["metric"] = plot_long["metric"].map(
        {
            "average_precision": "Average precision",
            "fake_precision": "Fake precision",
            "fake_recall": "Fake recall",
            "fake_f1": "Fake F1",
        }
    )
    plt.figure(figsize=(11, 6.5))
    sns.barplot(data=plot_long, y="model", x="score", hue="metric")
    plt.title("Shortcut Feature Audit")
    plt.xlabel("Holdout score")
    plt.ylabel("Feature set")
    plt.xlim(0, 1)
    _savefig(fig_dir, "shortcut_feature_audit_metrics")
    return shortcut_df


def _recompute_combined_text(x_frame):
    for col in TEXT_COLS:
        x_frame[col] = x_frame[col].fillna("")
    x_frame["combined_text"] = x_frame[TEXT_COLS].agg(" ".join, axis=1)
    return x_frame


def _word_count(series):
    return series.fillna("").astype(str).map(lambda value: len(value.split()))


def _make_remove_credibility_scenario(x_test):
    x_mod = x_test.copy()
    x_mod["company_profile"] = ""
    x_mod["benefits"] = ""
    x_mod["has_company_logo"] = 0
    x_mod["has_company_profile"] = 0
    x_mod["has_benefits"] = 0
    for col in [
        "company_profile_length",
        "benefits_length",
        "company_profile_word_count",
        "benefits_word_count",
    ]:
        x_mod[col] = 0
    return _recompute_combined_text(x_mod)


def _make_add_generic_credibility_scenario(x_test):
    x_mod = x_test.copy()
    generic_profile = (
        "Established organization with a clear company profile, hiring process, "
        "and verified business information."
    )
    generic_benefits = "Competitive benefits package, professional development, and employee support."

    missing_profile = x_mod["has_company_profile"].fillna(0).astype(int).eq(0)
    missing_benefits = x_mod["has_benefits"].fillna(0).astype(int).eq(0)
    x_mod.loc[missing_profile, "company_profile"] = generic_profile
    x_mod.loc[missing_benefits, "benefits"] = generic_benefits
    x_mod["has_company_logo"] = 1
    x_mod["has_company_profile"] = 1
    x_mod["has_benefits"] = 1
    x_mod["company_profile_length"] = x_mod["company_profile"].fillna("").astype(str).str.len()
    x_mod["benefits_length"] = x_mod["benefits"].fillna("").astype(str).str.len()
    x_mod["company_profile_word_count"] = _word_count(x_mod["company_profile"])
    x_mod["benefits_word_count"] = _word_count(x_mod["benefits"])
    return _recompute_combined_text(x_mod)


def _run_counterfactual_credibility_test(
    model, x_test, y_test, original_scores, table_dir, fig_dir, display_fn
):
    scenarios = {
        "original_holdout": x_test.copy(),
        "remove_profile_logo_benefits": _make_remove_credibility_scenario(x_test),
        "add_generic_profile_logo_benefits_to_sparse_rows": _make_add_generic_credibility_scenario(x_test),
    }

    rows = []
    shift_rows = []
    for scenario, x_values in scenarios.items():
        scores = model.decision_function(x_values)
        metrics = _metrics_from_scores(y_test, scores)
        metrics["scenario"] = scenario
        rows.append(metrics)

        for actual_class, class_label in [(0, "real"), (1, "fake")]:
            mask = y_test.to_numpy() == actual_class
            shift_rows.append(
                {
                    "scenario": scenario,
                    "actual_class": class_label,
                    "mean_original_score": float(np.mean(original_scores[mask])),
                    "mean_scenario_score": float(np.mean(scores[mask])),
                    "mean_score_shift": float(np.mean(scores[mask] - original_scores[mask])),
                    "predicted_fake_rate": float((scores[mask] >= 0).mean()),
                    "row_count": int(mask.sum()),
                }
            )

    scenario_df = pd.DataFrame(rows)
    scenario_cols = [
        "scenario",
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
    _display_and_save(
        scenario_df[scenario_cols].round(4),
        table_dir,
        "counterfactual_credibility_test",
        display_fn,
    )
    shift_df = pd.DataFrame(shift_rows)
    _display_and_save(
        shift_df.round(4),
        table_dir,
        "counterfactual_credibility_score_shifts",
        display_fn,
    )

    plot_long = scenario_df.melt(
        id_vars="scenario",
        value_vars=["predicted_fake_rate", "fake_precision", "fake_recall", "fake_f1"],
        var_name="metric",
        value_name="value",
    )
    plot_long["metric"] = plot_long["metric"].map(
        {
            "predicted_fake_rate": "Predicted fake rate",
            "fake_precision": "Fake precision",
            "fake_recall": "Fake recall",
            "fake_f1": "Fake F1",
        }
    )
    plt.figure(figsize=(11, 5.8))
    sns.barplot(data=plot_long, y="scenario", x="value", hue="metric")
    plt.title("Counterfactual Credibility Perturbation")
    plt.xlabel("Value")
    plt.ylabel("Scenario")
    plt.xlim(0, 1)
    _savefig(fig_dir, "counterfactual_credibility_metrics")

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=shift_df[shift_df["scenario"].ne("original_holdout")],
        x="mean_score_shift",
        y="scenario",
        hue="actual_class",
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Mean Score Shift From Credibility Counterfactuals")
    plt.xlabel("Mean shift in score for fake class")
    plt.ylabel("Scenario")
    _savefig(fig_dir, "counterfactual_credibility_score_shift")
    return scenario_df, shift_df


def _run_subgroup_robustness(
    x_test, y_test, scores, table_dir, fig_dir, display_fn, min_rows=50
):
    subgroup_features = [
        "employment_type",
        "required_experience",
        "required_education",
        "telecommuting",
        "has_company_logo",
        "has_company_profile",
        "has_salary_range",
        "has_benefits",
        "industry",
        "function",
    ]
    rows = []
    for feature in subgroup_features:
        values = x_test[feature].fillna("missing") if x_test[feature].dtype == object else x_test[feature]
        for value, idx in values.groupby(values).groups.items():
            idx = list(idx)
            if len(idx) < min_rows:
                continue
            row = _metrics_for_group(y_test.loc[idx], scores[x_test.index.get_indexer(idx)])
            row["feature"] = feature
            row["feature_value"] = value if value != "" else "missing"
            rows.append(row)

    subgroup_df = pd.DataFrame(rows)
    subgroup_df = subgroup_df.sort_values(["feature", "fake_recall", "row_count"], ascending=[True, True, False])
    _display_and_save(
        subgroup_df.round(4),
        table_dir,
        "subgroup_robustness_metrics",
        display_fn,
    )

    plot_df = subgroup_df[
        subgroup_df["feature"].isin(["has_company_logo", "has_company_profile", "employment_type", "required_experience"])
        & subgroup_df["fake_count"].ge(5)
    ].copy()
    plot_df["label"] = plot_df["feature"] + "=" + plot_df["feature_value"].astype(str)
    plot_df = plot_df.sort_values("fake_recall")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, y="label", x="fake_recall", color="#4c78a8")
    plt.title("Subgroup Robustness: Fake Recall by Group")
    plt.xlabel("Fake recall")
    plt.ylabel("Subgroup")
    plt.xlim(0, 1)
    _savefig(fig_dir, "subgroup_fake_recall_selected_groups")
    return subgroup_df


def _run_error_case_exports(x_test, y_test, scores, table_dir, display_fn):
    y_pred = (scores >= 0).astype(int)
    cases = x_test.copy()
    cases["actual"] = y_test.to_numpy()
    cases["predicted"] = y_pred
    cases["score_for_fake"] = scores
    cases["description_snippet"] = cases["description"].fillna("").astype(str).str.slice(0, 260)
    cols = [
        "job_id",
        "title",
        "location",
        "employment_type",
        "required_experience",
        "industry",
        "function",
        "has_company_logo",
        "has_company_profile",
        "has_benefits",
        "description_length",
        "company_profile_length",
        "benefits_length",
        "actual",
        "predicted",
        "score_for_fake",
        "description_snippet",
    ]

    false_positives = (
        cases[(cases["actual"].eq(0)) & (cases["predicted"].eq(1))]
        .sort_values("score_for_fake", ascending=False)
        [cols]
        .head(25)
    )
    false_negatives = (
        cases[(cases["actual"].eq(1)) & (cases["predicted"].eq(0))]
        .sort_values("score_for_fake", ascending=True)
        [cols]
        .head(25)
    )
    _display_and_save(
        false_positives.round(4),
        table_dir,
        "top_false_positive_case_studies",
        display_fn,
    )
    _display_and_save(
        false_negatives.round(4),
        table_dir,
        "top_false_negative_case_studies",
        display_fn,
    )
    return false_positives, false_negatives


def run_artifact_robustness_audit(
    data_path="Data/fake_jobs_cleaned.csv",
    output_dir="artifact_audit_outputs",
    display_fn=None,
    random_state=42,
):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    sns.set_theme(style="whitegrid", palette="Set2")

    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_frame(data_path)
    split_df, split_artifacts = _run_duplicate_and_split_audit(
        df, table_dir, fig_dir, display_fn, random_state
    )
    random_artifacts = split_artifacts["random_stratified_split"]

    shortcut_df = _run_shortcut_feature_audit(
        df,
        random_artifacts["train_idx"],
        random_artifacts["test_idx"],
        table_dir,
        fig_dir,
        display_fn,
        random_state,
    )
    counterfactual_df, counterfactual_shift_df = _run_counterfactual_credibility_test(
        random_artifacts["model"],
        random_artifacts["x_test"],
        random_artifacts["y_test"],
        random_artifacts["scores"],
        table_dir,
        fig_dir,
        display_fn,
    )
    subgroup_df = _run_subgroup_robustness(
        random_artifacts["x_test"],
        random_artifacts["y_test"],
        random_artifacts["scores"],
        table_dir,
        fig_dir,
        display_fn,
    )
    false_positives, false_negatives = _run_error_case_exports(
        random_artifacts["x_test"],
        random_artifacts["y_test"],
        random_artifacts["scores"],
        table_dir,
        display_fn,
    )

    return {
        "split_robustness_comparison": split_df,
        "shortcut_feature_audit": shortcut_df,
        "counterfactual_credibility_test": counterfactual_df,
        "counterfactual_credibility_score_shifts": counterfactual_shift_df,
        "subgroup_robustness_metrics": subgroup_df,
        "top_false_positive_case_studies": false_positives,
        "top_false_negative_case_studies": false_negatives,
    }
