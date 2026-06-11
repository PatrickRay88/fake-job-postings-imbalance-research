# Beyond Accuracy: Evaluating Classic Machine Learning Models on an Imbalanced Fake Job Posting Dataset

## Core Research Question

How does class imbalance affect model evaluation and model selection when detecting fraudulent job postings?

## Main Thesis

Fake job postings are rare in this dataset, so accuracy alone is a weak metric. A model can appear highly accurate by predicting almost every posting as real while still failing at the actual task: identifying fake jobs. Better evaluation requires minority-class metrics such as recall, precision, F1, balanced accuracy, ROC AUC, and average precision.

## Dataset Story

This project uses two versions of the same fake job postings dataset:

- `fake_job_postings.csv`: original Kaggle-style dataset.
- `fake_jobs_cleaned.csv`: cleaned and feature-engineered dataset with added length, word count, and missingness features.

The target variable is `fraudulent`, where:

- `0` means real posting.
- `1` means fake posting.

The dataset is highly imbalanced:

- Real postings: 17,014
- Fake postings: 866
- Fake rate: about 4.84%

Because fake postings are only about 4.84% of the dataset, a naive model that predicts every posting as real would achieve about 95.16% accuracy. This makes accuracy misleading as the main evaluation metric.

## Research Questions

1. How imbalanced is the dataset, and why does that matter?
2. Is accuracy misleading for fake job detection?
3. Which classic machine learning model performs best on the minority fake class?
4. Does the cleaned and feature-engineered dataset improve model performance compared with the original dataset?
5. Which evaluation metric gives the most useful view of model performance?
6. What tradeoff exists between catching fake jobs and falsely flagging real jobs?
7. Do class-weighted models perform better than unweighted models on minority-class metrics?
8. Which words or metadata features appear most predictive of fraudulent postings?
9. What kinds of postings are most likely to become false positives or false negatives?

## Hypotheses

- H1: Accuracy will overestimate model usefulness because of severe class imbalance.
- H2: Class-weighted linear models will outperform simple baselines on minority-class metrics.
- H3: Text-based models will outperform metadata-only models because fraudulent language patterns are embedded in job descriptions, requirements, and company profiles.
- H4: The cleaned dataset will improve performance slightly, but not dramatically, because TF-IDF text features already capture much of the useful signal.

## Methodology

1. Compare the original and cleaned datasets.
2. Perform exploratory data analysis on class balance, missingness, text length, and categorical fake rates.
3. Establish a dummy majority-class baseline.
4. Compare classic machine learning models using stratified 5-fold cross-validation.
5. Evaluate models with accuracy, balanced accuracy, ROC AUC, average precision, fake-class precision, fake-class recall, and fake-class F1.
6. Compare original vs cleaned dataset results.
7. Compare weighted vs unweighted linear models.
8. Plot precision-recall curves for the strongest model on each dataset.
9. Interpret top predictive features from the best linear model.
10. Inspect false positives and false negatives for practical error analysis.

## Recommended Model Framing

The goal is not simply to maximize accuracy. The goal is to identify fake postings well enough to support human review while avoiding an unacceptable number of false alarms. That makes average precision, fake-class recall, fake-class precision, and fake-class F1 more useful than accuracy alone.

## Expected Project Argument

This project shows that fake job posting detection is not just a classification problem, but an imbalanced classification problem. Accuracy alone hides poor minority-class performance. By using stratified cross-validation and metrics focused on the fake class, the project can evaluate which classic machine learning models are actually useful for detecting fraudulent job postings.

## Limitations

- The dataset may not represent all modern fake job postings.
- Text patterns can change over time as scammers adapt.
- A model could learn dataset-specific artifacts rather than general fraud signals.
- False positives may unfairly flag legitimate employers.
- A real deployment should support human review rather than automatically removing job posts.

## Final Project Angle

The strongest final angle is:

**Beyond Accuracy: Evaluating Classic Machine Learning Models on an Imbalanced Fake Job Posting Dataset**

The project should emphasize that the best model is not necessarily the one with the highest accuracy. The best model is the one that performs well on the rare fake-job class while keeping false positives at an acceptable level.
