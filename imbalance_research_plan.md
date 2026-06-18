# Imbalance-Focused Research Plan

This repository is a separate continuation of the fake job postings project. It builds on the completed EDA, modeling, and error analysis from the original project, but its research focus is narrower:

**How does class imbalance affect model evaluation, model selection, threshold choice, and error patterns in fake job posting detection?**

## Purpose of This Separate Repo

The original project repo remains the main class project deliverable being used for the team presentation and current analysis.

This repo is for a second, more research-focused version of the project. New experiments, notebooks, reports, and interpretation work related to class imbalance should be added here so the original project does not change underneath the rest of the team.

## Main Research Question

How does class imbalance affect model evaluation and decision-making in binary fake job posting classification?

## Subquestions

1. How misleading is accuracy when the fake class is rare?
2. Do different metrics select different best-performing models?
3. How does class weighting change fake-class precision and recall?
4. How does threshold choice change false positives and false negatives?
5. Which error patterns appear in false positives and false negatives?
6. Which feature conditions are associated with higher error rates?

## Existing Results This Repo Builds On

The current project already includes:

- class imbalance summary
- dummy majority baseline
- stratified 5-fold cross-validation
- classic model comparison
- original vs cleaned dataset comparison
- weighted vs unweighted model comparison
- precision-recall curves
- threshold analysis
- false positive / false negative review
- four-group error analysis
- binary-feature error rates
- text-length error rates
- score distribution by prediction group
- most confident mistakes and borderline cases

## Recommended Next Experiment

Add a **metric-based model selection table** that identifies which model appears best under each metric:

| Metric | What It Prioritizes |
|---|---|
| Accuracy | Overall correctness, strongly affected by class imbalance |
| Balanced accuracy | More equal treatment of real and fake classes |
| Fake precision | Fewer real postings incorrectly flagged as fake |
| Fake recall | More fake postings detected |
| Fake F1 | Balance between fake precision and fake recall |
| Average precision | Ranking quality for the rare fake class |

This experiment would directly test the idea that the definition of the "best" model changes depending on the metric used.

## Expected Research Claim

In imbalanced fake job posting detection, model performance depends heavily on the evaluation metric and decision threshold. Accuracy hides minority-class failure, while precision, recall, F1, average precision, threshold analysis, and error analysis reveal different tradeoffs.
