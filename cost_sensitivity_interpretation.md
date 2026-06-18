# Cost Sensitivity Analysis Interpretation

## Purpose of Cost Sensitivity Analysis

The threshold analysis showed that changing the decision threshold changes false positives and false negatives. Cost sensitivity analysis adds one more question:

**What threshold would be selected if false negatives were considered more costly than false positives?**

This analysis does not claim that the listed costs are the true real-world costs. Instead, it is a sensitivity analysis. It tests how the selected threshold changes under different assumptions about the relative cost of mistakes.

## Error Types Used in the Cost Calculation

The analysis uses two error types:

| Error Type | Meaning in This Project |
|---|---|
| False positive | A real job posting incorrectly predicted as fake |
| False negative | A fake job posting incorrectly predicted as real |

The cost sensitivity analysis keeps the false positive cost fixed at `1`. It then increases the false negative cost from `1` to `20`.

This allows comparison across scenarios:

- If false positives and false negatives are treated equally, both have cost `1`.
- If false negatives are treated as more serious, the false negative cost is increased.

## Cost Formula

For each threshold, the total cost is calculated as:

```text
total cost = (false positives * false positive cost) + (false negatives * false negative cost)
```

In this analysis:

```text
false positive cost = 1
false negative cost = 1, 2, 5, 10, or 20
```

The selected threshold is the threshold with the lowest total cost under that cost setting.

## Cost Sensitivity Results

| False Positive Cost | False Negative Cost | Selected Threshold | Flagged Rate | False Positives | False Negatives | Fake Precision | Fake Recall | Total Cost |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.2374 | 0.0400 | 46 | 196 | 0.9358 | 0.7737 | 242 |
| 1 | 2 | 0.0000 | 0.0471 | 115 | 139 | 0.8634 | 0.8395 | 393 |
| 1 | 5 | -0.0870 | 0.0500 | 152 | 124 | 0.8300 | 0.8568 | 772 |
| 1 | 10 | -0.5056 | 0.0750 | 534 | 59 | 0.6018 | 0.9319 | 1,124 |
| 1 | 20 | -0.5056 | 0.0750 | 534 | 59 | 0.6018 | 0.9319 | 1,714 |

Full table: [cost_sensitivity_threshold_selection.csv](imbalance_research_outputs/tables/cost_sensitivity_threshold_selection.csv)

## How to Read the Table

Each row represents a different assumption about the cost of missing a fake job posting.

For example, the first row treats false positives and false negatives equally:

```text
false positive cost = 1
false negative cost = 1
```

Under this assumption, the selected threshold is `0.2374`. At this threshold:

- 4% of postings are predicted as fake.
- There are 46 false positives.
- There are 196 false negatives.
- Fake precision is 0.9358.
- Fake recall is 0.7737.

This threshold is relatively conservative. It keeps false positives low but misses more fake postings.

## Why the Selected Threshold Changes

As false negatives become more costly, the selected threshold moves downward.

A lower threshold means the model predicts fake more often. This usually produces:

- more true positives
- fewer false negatives
- more false positives
- lower fake precision
- higher fake recall

This pattern appears in the cost sensitivity table.

When the false negative cost increases from `1` to `10`, the selected threshold changes from `0.2374` to `-0.5056`.

That shift changes the model behavior:

| Setting | Threshold | False Positives | False Negatives | Fake Precision | Fake Recall |
|---|---:|---:|---:|---:|---:|
| FN cost = 1 | 0.2374 | 46 | 196 | 0.9358 | 0.7737 |
| FN cost = 10 | -0.5056 | 534 | 59 | 0.6018 | 0.9319 |

When false negatives are cheap, the selected threshold avoids false positives. When false negatives are expensive, the selected threshold accepts more false positives in order to reduce missed fake postings.

## Plot: Selected Flagged Rate as False Negative Cost Increases

![Cost sensitivity selected flagged rate](imbalance_research_outputs/figures/cost_sensitivity_selected_flagged_rate.png)

### X-axis

The x-axis is the assumed cost of a false negative while the false positive cost stays fixed at `1`.

For example:

- `1` means a false negative and false positive are treated equally.
- `5` means a false negative is treated as five times as costly as a false positive.
- `10` means a false negative is treated as ten times as costly as a false positive.

### Y-axis

The y-axis is the selected flagged rate. This is the share of all postings predicted as fake at the threshold with the lowest total cost.

For example:

- A flagged rate of `0.0400` means 4% of postings are predicted as fake.
- A flagged rate of `0.0750` means 7.5% of postings are predicted as fake.

### Interpretation of the Plot

The selected flagged rate increases as the false negative cost increases.

This happens because higher false negative costs make missed fake postings more expensive in the total cost calculation. To reduce false negatives, the model must predict fake more often. Predicting fake more often requires a lower threshold, which increases the flagged rate.

The plot levels off at a flagged rate of `0.0750` for false negative costs of `10` and `20`. This happens because the tested thresholds are discrete. Among the thresholds tested, `-0.5056` gives the lowest total cost for both cost settings.

## Detailed Row-by-Row Interpretation

### False Negative Cost = 1

When false positives and false negatives are treated equally, the selected threshold is `0.2374`.

This threshold produces:

- 46 false positives
- 196 false negatives
- fake precision of 0.9358
- fake recall of 0.7737

The selected threshold is conservative because false positives and false negatives have equal cost, and the class imbalance means there are many real postings that could become false positives.

### False Negative Cost = 2

When false negatives are twice as costly as false positives, the selected threshold moves to the default threshold of `0.0000`.

This threshold produces:

- 115 false positives
- 139 false negatives
- fake precision of 0.8634
- fake recall of 0.8395

Compared with the equal-cost setting, the model accepts more false positives in exchange for fewer false negatives.

### False Negative Cost = 5

When false negatives are five times as costly as false positives, the selected threshold moves to `-0.0870`.

This threshold produces:

- 152 false positives
- 124 false negatives
- fake precision of 0.8300
- fake recall of 0.8568

The threshold is slightly more aggressive than the default threshold. It reduces false negatives but increases false positives.

### False Negative Cost = 10

When false negatives are ten times as costly as false positives, the selected threshold moves to `-0.5056`.

This threshold produces:

- 534 false positives
- 59 false negatives
- fake precision of 0.6018
- fake recall of 0.9319

This threshold is much more aggressive. It detects more fake postings but incorrectly flags many more real postings.

### False Negative Cost = 20

When false negatives are twenty times as costly as false positives, the selected threshold remains `-0.5056`.

This threshold again produces:

- 534 false positives
- 59 false negatives
- fake precision of 0.6018
- fake recall of 0.9319

The selected threshold does not change from the false-negative-cost-10 scenario because the tested threshold grid is limited. Among the available thresholds, `-0.5056` still minimizes total cost.

## Why This Analysis Matters for Imbalanced Data

Cost sensitivity analysis is important for imbalanced data because the two types of mistakes usually do not occur in equal numbers.

In this dataset, real postings greatly outnumber fake postings:

```text
Real postings: 17,014
Fake postings: 866
```

Because there are so many real postings, false positives can become large in count when the model predicts fake more aggressively. However, because fake postings are the minority class, missing fake postings can be hidden if evaluation focuses only on overall accuracy.

The cost sensitivity analysis makes this tradeoff explicit. It shows how the preferred threshold depends on whether the analysis treats missed fake postings as more costly than incorrectly flagged real postings.

## Limitations of This Analysis

This cost sensitivity analysis is simplified.

1. The cost values are hypothetical.
2. Only a small set of thresholds was tested.
3. The analysis does not estimate real financial, ethical, or operational costs.
4. The result depends on the selected model and dataset.
5. The analysis assumes the same cost for every false positive and the same cost for every false negative.

Because of these limitations, the results should be interpreted as a sensitivity test rather than a final cost model.

## Main Takeaway

The cost sensitivity analysis shows that model evaluation depends on assumptions about mistake costs.

- When false positives and false negatives are treated equally, the selected threshold is conservative and has high fake precision.
- When false negatives are treated as more costly, the selected threshold becomes more aggressive and has higher fake recall.
- As the false negative cost increases, the model flags a larger share of postings as fake.

This supports the broader research claim: in imbalanced classification, model performance is not fully described by one score. Metric choice, threshold choice, and error-cost assumptions all affect the final interpretation.
