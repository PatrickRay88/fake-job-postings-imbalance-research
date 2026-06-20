# Benchmark Audit Paper Outline

## Working Title

**Beyond Accuracy on EMSCAD: A Robustness and Shortcut Audit of Fake Job Posting Detection**

## One-Sentence Thesis

High fake-job detection performance on EMSCAD should be interpreted cautiously because model conclusions change under near-duplicate-aware splitting, order-based splitting, credibility metadata perturbations, fake-posting archetype, class imbalance settings, and review-policy assumptions.

## Abstract Draft

Online recruitment fraud detection is commonly evaluated as a binary classification problem using the EMSCAD fake job postings dataset. Although many studies report strong performance on this benchmark, less attention has been paid to whether those results remain stable under repeated templates, benchmark artifacts, and realistic decision constraints. This study audits EMSCAD as an evaluation benchmark rather than proposing a new classifier. We reproduce a strong classical machine learning baseline, then evaluate exact duplicate leakage, near-duplicate/template leakage, random versus grouped and order-based splits, shortcut feature reliance, counterfactual company-credibility metadata edits, fake-posting archetypes, class imbalance sensitivity, review-budget constraints, cost-sensitive thresholding, feature-group ablations, label scarcity, and subgroup error patterns. Results show that the classifier learns useful text signal, but reported performance is sensitive to near-duplicate-aware splitting and credibility metadata. In a random stratified split, 13.8% of test rows share exact content signatures with training; under a 0.98 text-similarity threshold, 50.72% of test rows share a near-duplicate/template cluster with training. Near-duplicate group splitting reduces fake F1 from 0.8905 to 0.7208. Counterfactual removal of company profile, logo, and benefits information increases false positives from 17 to 143, while adding generic credibility information reduces false positives but increases missed fake postings. Fake-posting archetype analysis shows that broad professional fake postings are harder to detect than smaller template-like scam archetypes. These findings suggest that EMSCAD remains useful, but benchmark reporting should include near-duplicate-aware splits, shortcut audits, archetype-level recall, and threshold-policy analysis before claims of deployment reliability are made.

## Contributions

1. **Benchmark audit framing:** treats EMSCAD as a benchmark to evaluate, not just a dataset to classify.
2. **Duplicate leakage analysis:** quantifies exact content duplication and random train-test overlap.
3. **Near-duplicate/template leakage analysis:** quantifies high-similarity posting clusters and random split overlap.
4. **Split robustness comparison:** compares random stratified, exact-duplicate-group, near-duplicate-group, and job-id-order split protocols.
5. **Shortcut feature audit:** tests whether flags, missingness, and text-length features alone can explain performance.
6. **Counterfactual credibility test:** measures how model decisions change when company profile/logo/benefits information is removed or added.
7. **Fake-archetype analysis:** identifies exploratory fake-posting topics and measures recall by archetype.
8. **Imbalance decision analysis:** evaluates prevalence, review budget, cost-sensitive thresholding, training balance, and label scarcity.
9. **Practical reporting recommendation:** proposes that future EMSCAD work report near-duplicate-aware splits, shortcut robustness, archetype-level recall, threshold behavior, and false-positive review risk.

## Research Questions

| Research Question | Experiment |
|---|---|
| RQ1: Does EMSCAD contain exact duplicate content that can leak across random splits? | Duplicate signature audit and random split leakage summary |
| RQ2: How stable is model performance under alternative split strategies? | Random stratified vs duplicate-group vs job-id-order split comparison |
| RQ3: Does near-duplicate/template leakage have a stronger effect than exact duplicate leakage? | TF-IDF/SVD near-duplicate clustering and near-duplicate group split |
| RQ4: Can shallow shortcut features explain classifier performance? | Shortcut-only, credibility-only, length-only, text-only, and all-feature models |
| RQ5: How sensitive are predictions to company credibility metadata? | Counterfactual removal/addition of company profile, logo, and benefits information |
| RQ6: Are some fake-posting archetypes easier or harder to detect? | NMF fake-posting archetypes and recall by archetype |
| RQ7: How do imbalance and threshold decisions affect interpretation? | Prevalence stress test, review budget analysis, cost sensitivity, threshold sensitivity |
| RQ8: What kinds of errors remain? | Subgroup robustness and false-positive/false-negative case exports |

## Methods Overview

### Dataset

- EMSCAD / Kaggle fake job postings dataset
- 17,880 postings
- 17,014 real postings
- 866 fake postings
- Fake rate: 4.84%

### Baseline Model

- Balanced Linear SVM
- TF-IDF text features from title, company profile, description, requirements, and benefits
- One-hot categorical metadata
- Binary credibility/presence indicators
- Length and word-count features

### Evaluation Metrics

- Accuracy
- Balanced accuracy
- ROC AUC
- Average precision
- Fake-class precision
- Fake-class recall
- Fake-class F1
- False positives and false negatives
- Predicted fake rate

## Experiment Plan

### 1. Reproduce Strong Baseline

Purpose: show that the model achieves strong conventional performance.

Key result:

- Fake F1: 0.8905
- Fake precision: 0.9167
- Fake recall: 0.8657

### 2. Duplicate Leakage Audit

Purpose: test whether random splitting exposes the model to repeated content across train and test.

Key result:

- 2,755 rows are in exact duplicate-content groups.
- 13.8% of random test rows share exact content signatures with training.

### 3. Split Robustness

Purpose: test whether performance changes under stricter evaluation protocols.

Key result:

- Random split fake F1: 0.8905
- Duplicate-group split fake F1: 0.8786
- Job-id-order split fake F1: 0.7915
- Job-id-order split fake recall: 0.6789

Interpretation:

Duplicate leakage contributes some optimism, but order-based shift has a larger effect.

### 4. Near-Duplicate / Template Leakage

Purpose: test whether random split performance relies on highly similar posting templates, not only exact duplicate rows.

Key result:

- At a 0.98 similarity threshold, 9,581 rows fall into near-duplicate/template clusters.
- 50.72% of random test rows share a near-duplicate/template cluster with training.
- Near-duplicate group split fake F1: 0.7208.
- Near-duplicate group split fake recall: 0.6368.

Interpretation:

Template-aware splitting produces a much larger drop than exact duplicate grouping, suggesting EMSCAD benchmark performance is partly shaped by repeated posting language.

### 5. Shortcut Feature Audit

Purpose: determine whether simple missingness/length/credibility flags explain performance.

Key result:

- All features fake F1: 0.8905
- Text only fake F1: 0.8498
- Shortcut features only fake F1: 0.2726
- Length only fake F1: 0.1643

Interpretation:

The classifier is not only a shortcut model. Text carries substantial signal.

### 6. Counterfactual Credibility Audit

Purpose: test whether company profile/logo/benefits information changes predictions.

Key result:

- Original false positives: 17
- False positives after removing profile/logo/benefits: 143
- False positives after adding generic credibility information: 4
- False negatives after adding generic credibility information: 54

Interpretation:

Credibility metadata strongly influences decision behavior, even though shortcut features alone are not enough.

### 7. Fake-Posting Archetype Analysis

Purpose: test whether fake postings form subtypes with different detection difficulty.

Key result:

- Six exploratory fake-posting archetypes were identified.
- The broad professional/business archetype has the lowest recall: 0.8344.
- Smaller template-like archetypes, including cruise/service and home-office postings, reached 1.0000 recall in the holdout split.

Interpretation:

Aggregate fake-class recall hides heterogeneity in the fake class. The model detects distinctive scam templates more easily than broad fake postings that resemble regular professional listings.

### 8. Imbalance Decision Analysis

Purpose: show that model interpretation changes under deployment-like assumptions.

Includes:

- Prevalence stress test
- Review-budget analysis
- Threshold sensitivity
- Expanded cost sensitivity
- Training distribution experiment
- Label scarcity experiment

Key results:

- Fake precision changes from 0.6908 at 1% fake prevalence to 0.9819 at 20% fake prevalence.
- Reviewing the top 5% highest-risk postings catches 88.4% of fake postings but reviews 33 real postings.
- Increasing false-negative cost selects lower thresholds and increases false positives.

### 9. Subgroup and Error Analysis

Purpose: identify where the model is less reliable.

Key result:

- Rows without company profile/logo have higher fake rates but also higher false-positive risk among real postings.
- Sparse legitimate postings are a major false-positive failure mode.

## Proposed Paper Structure

1. Introduction
2. Background and Related Work
3. Dataset and Baseline Model
4. Research Questions
5. Experimental Design
6. Results
   - Baseline performance
   - Duplicate leakage
   - Split robustness
   - Shortcut feature audit
   - Counterfactual credibility audit
   - Imbalance decision analysis
   - Subgroup/error analysis
7. Discussion
8. Limitations
9. Benchmark Reporting Recommendations
10. Conclusion

## Benchmark Reporting Recommendations

Future EMSCAD papers should report:

1. Random split results and duplicate-group split results.
2. Whether exact or near-duplicate content appears across train/test.
3. Near-duplicate/template group split results.
4. Fake-class precision, recall, F1, average precision, and predicted fake rate.
5. Threshold sensitivity or review-budget analysis.
6. Shortcut-only baseline results.
7. Performance with and without credibility-related metadata.
8. Fake-archetype recall or fraud-type recall when labels are available.
9. Subgroup false-positive rates for sparse legitimate postings.

## Claim Wording For Final Report

Use this:

**This study does not claim that EMSCAD is invalid or that fake job detection cannot be modeled. Instead, it shows that EMSCAD performance should be interpreted as benchmark-specific unless models are evaluated with leakage-aware splits, shortcut audits, and threshold-policy analysis.**

Avoid this:

**The model detects fake jobs in the real world.**

## Limitations To State Clearly

- Exact duplicate signatures do not capture semantic near-duplicates.
- Job ID order is only a proxy for temporal or platform shift.
- Counterfactual edits are synthetic.
- EMSCAD is old and platform-specific.
- The project does not add external company registry or network data.
- The project does not annotate fraud subtypes.

## Final Publishable-Style Claim

**EMSCAD remains useful for studying online recruitment fraud, but conventional random-split classifier scores are incomplete. A benchmark audit reveals near-duplicate/template leakage, split sensitivity, fake-archetype heterogeneity, and credibility-metadata dependence that should be reported before using EMSCAD performance as evidence of deployment-ready fake job detection.**
