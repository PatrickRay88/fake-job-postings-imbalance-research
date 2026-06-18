# Threshold Sensitivity Interpretation

## Purpose of Threshold Analysis

The balanced Linear SVM produces a decision score for each job posting. The default decision threshold is `0.0`:

- Scores greater than or equal to `0.0` are predicted as fake.
- Scores below `0.0` are predicted as real.

Threshold analysis changes this cutoff and measures how the model's predictions change. This is useful for imbalanced classification because the default threshold is only one possible operating point. A different threshold can make the model more aggressive or more conservative when predicting the minority fake class.

## Threshold Table

| Threshold | Share Predicted Fake | True Positives | False Positives | False Negatives | Fake Precision | Fake Recall | Fake F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| -0.9649 | 0.1500 | 839 | 1,843 | 27 | 0.3128 | 0.9688 | 0.4729 |
| -0.7289 | 0.1000 | 823 | 965 | 43 | 0.4603 | 0.9503 | 0.6202 |
| -0.5056 | 0.0750 | 807 | 534 | 59 | 0.6018 | 0.9319 | 0.7313 |
| -0.0870 | 0.0500 | 742 | 152 | 124 | 0.8300 | 0.8568 | 0.8432 |
| 0.0000 | 0.0471 | 727 | 115 | 139 | 0.8634 | 0.8395 | 0.8513 |
| 0.2374 | 0.0400 | 670 | 46 | 196 | 0.9358 | 0.7737 | 0.8470 |
| 0.6742 | 0.0300 | 527 | 10 | 339 | 0.9814 | 0.6085 | 0.7512 |

Full table: [threshold_sensitivity_summary.csv](imbalance_research_outputs/tables/threshold_sensitivity_summary.csv)

## Separate Metric Plots

The notebook now separates fake precision, fake recall, and fake F1 into individual plots. This is easier to interpret than one combined plot because each metric answers a different question.

### Shared X-axis for the Three Metric Plots

The x-axis in each metric plot is the **share of postings predicted as fake**.

For example:

- `0.03` means the model flagged about 3% of all postings as fake.
- `0.05` means the model flagged about 5% of all postings as fake.
- `0.15` means the model flagged about 15% of all postings as fake.

Because the actual fake rate is about 4.84%, a flagged rate near 0.05 is close to the actual class distribution. A flagged rate much higher than 0.05 means the model is predicting fake more often than fake postings actually occur in the dataset.

### Plot 1: Fake Precision

![Threshold sensitivity fake precision](imbalance_research_outputs/figures/threshold_sensitivity_fake_precision_separate.png)

Fake precision measures:

```text
Of all postings predicted as fake, how many were actually fake?
```

The y-axis is fake precision. Higher values mean the predicted-fake group contains a larger share of actual fake postings.

When the flagged rate is low, the model only predicts fake for postings with stronger fake-class scores. This produces high precision. At the 3% flagged rate, fake precision is `0.9814`, meaning almost all postings predicted as fake were actually fake.

As the flagged rate increases, the model includes more borderline postings in the fake category. Many of those additional postings are real. This causes fake precision to decrease. At the 15% flagged rate, fake precision falls to `0.3128`.

### Plot 2: Fake Recall

![Threshold sensitivity fake recall](imbalance_research_outputs/figures/threshold_sensitivity_fake_recall_separate.png)

Fake recall measures:

```text
Of all actual fake postings, how many did the model detect?
```

The y-axis is fake recall. Higher values mean the model found a larger share of all actual fake postings.

When the flagged rate is low, the model misses many fake postings because it is only flagging the highest-scoring cases. At the 3% flagged rate, fake recall is `0.6085`.

As the flagged rate increases, the model predicts more postings as fake, so it captures more of the actual fake postings. At the 15% flagged rate, fake recall rises to `0.9688`.

### Plot 3: Fake F1

![Threshold sensitivity fake F1](imbalance_research_outputs/figures/threshold_sensitivity_fake_f1_separate.png)

Fake F1 combines fake precision and fake recall into one score. It is highest when precision and recall are balanced.

The y-axis is fake F1. A higher value means the threshold is balancing fake precision and fake recall more effectively.

In this analysis, F1 is strongest near the default threshold and nearby flagged rates:

- At the default threshold `0.0000`, fake F1 is `0.8513`.
- At a 4% flagged rate, fake F1 is `0.8470`.
- At a 5% flagged rate, fake F1 is `0.8432`.

F1 drops when the threshold becomes too conservative or too aggressive:

- At the 3% flagged rate, recall becomes low, so F1 drops to `0.7512`.
- At the 15% flagged rate, precision becomes low, so F1 drops to `0.4729`.

### Panel Comparison

![Separate threshold metric panels](imbalance_research_outputs/figures/threshold_sensitivity_separate_metric_panels.png)

The panel comparison uses the same x-axis for all three metrics but separates the y-axis view by metric. This makes it easier to see that precision decreases, recall increases, and F1 peaks near the middle.

## Plot 2: Threshold Sensitivity Outcomes

![Threshold sensitivity outcomes](imbalance_research_outputs/figures/threshold_sensitivity_outcomes.png)

This plot shows counts rather than metric scores.

### X-axis

The x-axis is again the **share of postings predicted as fake**.

Moving left to right means the model is flagging a larger portion of all postings as fake.

### Y-axis

The y-axis is the **number of postings** in each outcome group:

- true positives
- false positives
- false negatives

### True Positives

True positives are fake postings correctly predicted as fake.

As the share predicted as fake increases, true positives increase:

- 527 true positives at the 3% flagged rate
- 727 true positives at the default threshold
- 839 true positives at the 15% flagged rate

This happens because a lower threshold allows the model to catch more of the actual fake postings.

### False Negatives

False negatives are fake postings incorrectly predicted as real.

As the share predicted as fake increases, false negatives decrease:

- 339 false negatives at the 3% flagged rate
- 139 false negatives at the default threshold
- 27 false negatives at the 15% flagged rate

This is the inverse of true positives. When more fake postings are captured, fewer fake postings are missed.

### False Positives

False positives are real postings incorrectly predicted as fake.

This line rises sharply as the share predicted as fake increases:

- 10 false positives at the 3% flagged rate
- 115 false positives at the default threshold
- 1,843 false positives at the 15% flagged rate

The false-positive line rises much higher than the true-positive line because the dataset contains far more real postings than fake postings:

```text
Real postings: 17,014
Fake postings: 866
```

There are about 19.6 real postings for every fake posting. Because the real class is so much larger, even a small percentage of real postings being incorrectly flagged can create a large number of false positives.

This explains why the false-positive count grows so quickly when the threshold becomes more aggressive. The model is not necessarily making a higher error rate on real postings than on fake postings; rather, the real class is much larger, so false positives can dominate the count plot once the model starts flagging more postings.

## Why the False Positive Line Rises So Much

The key reason is class imbalance.

At the 15% flagged rate, the model predicts 2,682 postings as fake:

```text
839 true positives + 1,843 false positives = 2,682 predicted fake postings
```

But the dataset only has 866 actual fake postings. Once the model flags far more postings than the actual number of fake postings, many of the additional flagged postings must come from the real class.

This is why false positives rise quickly at higher flagged rates. The model is expanding the fake prediction group into a much larger real-posting population.

## Interpretation of the Default Threshold

At the default threshold of `0.0000`, the model predicts about 4.71% of postings as fake. This is close to the actual fake rate of 4.84%.

At this threshold, the model produces:

| Outcome | Count |
|---|---:|
| True positives | 727 |
| False positives | 115 |
| False negatives | 139 |

The corresponding fake-class metrics are:

| Metric | Value |
|---|---:|
| Fake precision | 0.8634 |
| Fake recall | 0.8395 |
| Fake F1 | 0.8513 |

This threshold gives a relatively balanced precision-recall result for this model. It does not maximize recall, and it does not maximize precision. Instead, it provides a middle point between detecting fake postings and limiting false positives.

## Conservative vs Aggressive Thresholds

### Conservative Threshold

The 3% flagged rate is more conservative:

| Metric | Value |
|---|---:|
| Fake precision | 0.9814 |
| Fake recall | 0.6085 |
| False positives | 10 |
| False negatives | 339 |

This setting produces very few false positives, but it misses many fake postings.

### Aggressive Threshold

The 15% flagged rate is more aggressive:

| Metric | Value |
|---|---:|
| Fake precision | 0.3128 |
| Fake recall | 0.9688 |
| False positives | 1,843 |
| False negatives | 27 |

This setting detects almost all fake postings, but it incorrectly flags many real postings as fake.

## Main Threshold Takeaway

The threshold plots show that the same model can behave very differently depending on the decision threshold.

- Lower threshold: more fake postings detected, more real postings incorrectly flagged.
- Higher threshold: fewer real postings incorrectly flagged, more fake postings missed.

Because the dataset is highly imbalanced, false positives can become large in count when the model predicts a larger share of postings as fake. This happens because real postings greatly outnumber fake postings.

The threshold analysis therefore supports the main research point: in imbalanced classification, model evaluation is not only about the algorithm. It also depends on the metric and threshold used to convert model scores into final predictions.
