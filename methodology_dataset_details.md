# Methodology and Dataset Details

## Cross-Validation Setup

The project used **stratified 5-fold cross-validation**.

In the notebook, the cross-validation object was defined as:

```python
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

This means the dataset was split into 5 folds. Each model was trained on 4 folds and tested on the remaining fold. This process repeated 5 times so that each row appeared in a test fold once.

The cross-validation used `shuffle=True` to randomize fold assignment and `random_state=42` to make the results reproducible. The use of `StratifiedKFold` was important because the dataset is imbalanced. Stratification preserves the approximate real/fake class ratio inside each fold, which helps ensure that each fold contains examples from the minority fake class.

## Dataset Description

The project used two versions of the same fake job postings dataset.

| Dataset | File | Rows | Columns |
|---|---|---:|---:|
| Original dataset | `Data/fake_job_postings.csv` | 17,880 | 18 |
| Cleaned dataset | `Data/fake_jobs_cleaned.csv` | 17,880 | 32 |

The target variable is:

```text
fraudulent
```

The target values are:

| Value | Meaning |
|---:|---|
| 0 | Real job posting |
| 1 | Fake/fraudulent job posting |

The class distribution is:

| Class | Count | Percent |
|---|---:|---:|
| Real postings | 17,014 | 95.16% |
| Fake postings | 866 | 4.84% |

This class distribution shows that the dataset is highly imbalanced. Most postings are real, while only a small minority are fake.

## Original Dataset Features

The original dataset includes the following columns:

```text
job_id
title
location
department
salary_range
company_profile
description
requirements
benefits
telecommuting
has_company_logo
has_questions
employment_type
required_experience
required_education
industry
function
fraudulent
```

### Text Features

The main text fields are:

```text
title
company_profile
description
requirements
benefits
```

These fields contain the written content of the job posting. They were used to build TF-IDF features for the text-based models.

### Categorical Features

The categorical fields are:

```text
location
department
employment_type
required_experience
required_education
industry
function
```

These features describe the job posting using categories such as job location, employment type, required experience level, industry, and job function.

### Binary Indicator Features

The original binary indicator fields are:

```text
telecommuting
has_company_logo
has_questions
```

These variables indicate whether a posting is remote/telecommuting, whether it includes a company logo, and whether it includes screening questions.

### Identifier

The `job_id` column is an identifier. It was not used as a predictive modeling feature.

## Cleaned Dataset Added Features

The cleaned dataset contains the original fields plus engineered features.

### Text Length Features

```text
title_length
description_length
requirements_length
company_profile_length
benefits_length
```

These features measure the number of characters in each major text field.

### Missingness and Presence Indicators

```text
has_salary_range
has_benefits
has_company_profile
has_department
```

These features indicate whether important fields are present or missing.

### Word Count Features

```text
title_word_count
description_word_count
requirements_word_count
company_profile_word_count
benefits_word_count
```

These features measure the number of words in each major text field.

The engineered features help describe the structure and completeness of job postings. For example, they allow analysis of whether fake postings tend to have shorter descriptions, missing company profiles, or less detailed requirements.

## Preprocessing Pipeline

The notebook combined the main text fields into one text feature:

```text
title + company_profile + description + requirements + benefits
```

The combined text was transformed using TF-IDF:

```python
TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=6000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)
```

This configuration used:

- lowercase text normalization
- English stopword removal
- up to 6,000 text features
- unigrams and bigrams
- terms appearing in at least 2 documents
- sublinear term-frequency scaling

Categorical features were imputed with the value `"missing"` and then one-hot encoded.

Numeric features were imputed with median values and scaled with `MaxAbsScaler`.

## Models Compared

The main model comparison included:

```text
Dummy majority baseline
Logistic Regression balanced
Linear SVM balanced
Complement Naive Bayes text only
Random Forest metadata only
```

The dummy majority baseline was included to show why accuracy is misleading. Since most postings are real, a model can achieve high accuracy by predicting every posting as real while detecting no fake postings.

## Evaluation Metrics

The notebook reported the following metrics:

```text
accuracy
balanced accuracy
ROC AUC
average precision
fake-class precision
fake-class recall
fake-class F1
```

The primary model comparison metric was:

```text
average precision
```

Average precision was used because the fake class is rare. Accuracy was included, but it was not used as the main metric because the majority real class strongly affects accuracy.

## Selected Model

The selected model for detailed analysis was:

```text
Balanced Linear SVM on the cleaned dataset
```

This model was selected because it had the highest average precision in the main cleaned-dataset model comparison while maintaining strong fake-class precision and recall.

The cleaned dataset balanced Linear SVM results were:

| Metric | Value |
|---|---:|
| Accuracy | 0.9858 |
| Balanced accuracy | 0.9164 |
| ROC AUC | 0.9890 |
| Average precision | 0.9140 |
| Fake precision | 0.8636 |
| Fake recall | 0.8395 |
| Fake F1 | 0.8512 |

## Interpretation

The project is not only a binary classification task. It is an imbalanced classification task.

Because only 4.84% of the postings are fake, accuracy alone does not adequately evaluate model performance. A model can predict most postings as real and still achieve high accuracy. For this reason, the analysis emphasizes average precision, fake-class recall, fake-class precision, fake-class F1, confusion matrices, threshold tradeoffs, and error analysis.

The main evaluation question is whether the model can detect the minority fake class while controlling false positives among real postings.
