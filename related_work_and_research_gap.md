# Related Work and Research Gap

## Proposed Research Title

**Beyond Accuracy on EMSCAD: A Robustness and Shortcut Audit of Fake Job Posting Detection**

## Core Research Gap

The EMSCAD / fake job postings dataset has already been used many times for binary fake-vs-real classification. A new project is unlikely to be publishable if it only reports another model comparison or another accuracy score.

The stronger gap is benchmark trustworthiness:

**Does EMSCAD still provide a reliable benchmark for fake job detection, or do duplicate content, random split design, class imbalance, and credibility metadata artifacts inflate reported model performance?**

This project responds to that gap by treating EMSCAD as a benchmark to audit, not just a dataset to classify.

## Literature-to-Project Gap Table

| Source | What Prior Work Did | Future Work / Limitation Direction | How This Project Responds |
|---|---|---|---|
| Vidros et al. (2017), EMSCAD | Introduced EMSCAD and benchmarked online recruitment fraud detection using job text and metadata. | Future work called for expanding EMSCAD, adding user behavior, company/network data, user-content-IP collision patterns, and graph modeling between jobs, companies, and users. | We cannot add unavailable network/user data, but we audit the benchmark using exact content signatures, duplicate leakage checks, split robustness, and company credibility counterfactuals. This is a partial response to the call for richer structural/contextual evaluation. |
| Vo et al. (2021), class imbalance | Focused on imbalance-aware learning and oversampling for fake job description detection. | Future work points toward better imbalance methods, feature extraction, and online/evolving data settings. | We add threshold sensitivity, review-budget analysis, training-distribution comparison, label-scarcity analysis, and a job-id-order split as a proxy for distribution shift over the dataset order. |
| Mahbub, Pardede, and Kayes (2022), contextual features | Studied contextual features in Australian job-market ORF detection and emphasized localization/context. | The work suggests that detection should use context beyond generic text features and that local market context affects generalization. | We add subgroup robustness and credibility-metadata counterfactuals. We also identify a remaining gap: EMSCAD lacks external company registry, domain, and local labor-market context. |
| Adebayo et al., fraudulent job types | Moved beyond binary classification by classifying fraudulent job types. | Future directions include richer public datasets and more current job advertisements. | We do not annotate fraud types, but our audit complements this direction by showing that binary EMSCAD performance should be interpreted cautiously before being generalized. |
| Alghamdi and Alharby (2019), ORF detection | Built an ensemble model for ORF detection and emphasized attributes such as company profile, logo, and required experience. | Future work suggests continued feature study and broader datasets/settings. | We directly test company profile/logo/benefits sensitivity using counterfactual edits and subgroup analysis. |

## What Is New in This Project

The project does not claim to invent a new classifier. Its contribution is an evaluation audit.

Main contributions:

1. **Duplicate leakage audit:** identifies exact content signatures and measures how much random test data overlaps with training content.
2. **Near-duplicate / template leakage audit:** uses text similarity to identify repeated posting templates beyond exact duplicates.
3. **Split robustness audit:** compares random stratified, exact-duplicate-group, near-duplicate-group, and job-id-order splits.
4. **Shortcut feature audit:** tests whether missingness, length, and credibility flags alone can explain model performance.
5. **Counterfactual credibility audit:** removes or adds company profile/logo/benefits information and measures score and error changes.
6. **Fake-posting archetype analysis:** uses unsupervised topic modeling to identify fake-posting subtypes and measure recall by subtype.
7. **Imbalance decision audit:** evaluates threshold choice, review budgets, cost sensitivity, prevalence stress, training balance, and label scarcity.
8. **Subgroup and error analysis:** examines where false positives and false negatives occur.

## Why This Is a More Publishable Direction

Most EMSCAD projects ask:

**Which model classifies fake jobs best?**

This project asks:

**When should we trust the reported performance of fake job classifiers on EMSCAD?**

That is a stronger research question because it evaluates the benchmark itself. The result can be useful even if no new classifier is proposed.

## Revised Research Questions

1. **RQ1:** Does random splitting on EMSCAD produce duplicate-content leakage between train and test sets?
2. **RQ2:** How much does performance change under duplicate-group and job-id-order splits?
3. **RQ3:** Does near-duplicate or template-level leakage have a stronger effect than exact duplicate leakage?
4. **RQ4:** Can shortcut features such as missingness, length, and credibility flags explain model performance?
5. **RQ5:** How sensitive are model predictions to counterfactual changes in company profile, logo, and benefits information?
6. **RQ6:** Are some fake-posting archetypes easier or harder to detect than others?
7. **RQ7:** How do class imbalance, threshold policy, review budget, and cost assumptions change model interpretation?
8. **RQ8:** Which subgroups and error types reveal the strongest limits of the classifier?

## Current Evidence From This Project

The current audit found:

- 2,755 rows belong to exact duplicate-content groups.
- 13.8% of random test rows have an exact content signature that appears in training.
- Duplicate-group splitting lowers fake F1 from 0.8905 to 0.8786.
- At a 0.98 text-similarity threshold, 9,581 rows belong to near-duplicate/template clusters.
- 50.72% of random-split test rows share a near-duplicate/template cluster with training.
- Near-duplicate group splitting lowers fake F1 from 0.8905 to 0.7208.
- Job-id-order splitting lowers fake recall from 0.8657 to 0.6789.
- Shortcut-only features perform poorly, so the classifier is not explained by missingness and length alone.
- Removing company profile/logo/benefits increases false positives from 17 to 143.
- Adding generic credibility information lowers false positives from 17 to 4 but increases missed fake postings from 29 to 54.
- The broad professional fake-posting archetype has lower recall than smaller, more template-like fake archetypes.

These results support a nuanced claim:

**The model has real signal, especially from text, but EMSCAD performance is sensitive to repeated templates, split strategy, credibility metadata, and fake-posting subtype.**

## Claim Boundaries

This project should not claim:

- That EMSCAD is invalid.
- That fake job detection is solved.
- That the model would work in real deployment.
- That company profile/logo/benefits are inherently fair or reliable fraud indicators.

This project can claim:

- EMSCAD contains duplicate content that can leak across random splits.
- Random split performance is more optimistic than some alternative split strategies.
- Near-duplicate/template-aware splitting produces a much larger performance drop than exact-duplicate splitting.
- The classifier is not purely a shortcut model.
- Credibility metadata strongly affects model behavior.
- Benchmark reporting should include leakage-aware splits, near-duplicate-aware splits, threshold analysis, fake-archetype recall, and shortcut robustness checks.

## Remaining Future Work

The strongest next steps would be:

1. **Near-duplicate detection:** use semantic similarity or MinHash rather than exact content signatures.
2. **External validation:** test on a newer fake-job dataset or collect a small modern validation set.
3. **Temporal validation:** if true posting dates can be recovered, use time-based splits instead of job-id order.
4. **Contextual verification:** add company-domain, company-registry, website, and social-footprint features.
5. **Graph modeling:** model relationships between companies, job posts, locations, emails, and repeated content.
6. **Fraud-type labels:** annotate fake jobs into scam categories rather than treating all fake postings as one class.
7. **Fairness and false-positive review:** study whether sparse legitimate postings from small organizations are disproportionately flagged.

## References

- Vidros, S., Kolias, C., Kambourakis, G., and Akoglu, L. (2017). Automatic Detection of Online Recruitment Frauds: Characteristics, Methods, and a Public Dataset. [Future Internet](https://www.mdpi.com/1999-5903/9/1/6).
- Vo et al. (2021). Dealing with the Class Imbalance Problem in the Detection of Fake Job Descriptions. [Computers, Materials & Continua](https://www.techscience.com/cmc/v68n1/41824/html).
- Mahbub, S., Pardede, E., and Kayes, A. S. M. (2022). Online Recruitment Fraud Detection: A Study on Contextual Features in Australian Job Industries. [IEEE Access / ResearchGate record](https://www.researchgate.net/publication/362580914_Online_Recruitment_Fraud_Detection_A_Study_on_Contextual_Features_in_Australian_Job_Industries).
- Adebayo et al. A machine learning approach to detecting fraudulent job types. [AI & Society](https://link.springer.com/article/10.1007/s00146-022-01469-0).
- Alghamdi, B., and Alharby, F. (2019). An Intelligent Model for Online Recruitment Fraud Detection. [Journal of Information Security](https://www.scirp.org/journal/paperinformation?paperid=93637).
