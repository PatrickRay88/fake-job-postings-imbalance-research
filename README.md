# Fake Job Postings ML Project

Exploratory data analysis and classic machine learning model comparison for an imbalanced fake job postings dataset.

## Separate Imbalance Research Focus

This repository is a separate continuation of the original class project. The added research focus is:

**How does class imbalance affect model evaluation, model selection, threshold choice, and error patterns in fake job posting detection?**

Start with:

- `related_work_and_research_gap.md`
- `benchmark_audit_paper_outline.md`
- `imbalance_research_plan.md`
- `imbalance_focused_research.ipynb`
- `future_work_extension_report.md`
- `artifact_robustness_audit_report.md`
- `imbalance_research_summary.md`
- `comprehensive_imbalance_experiment_report.md`
- `threshold_interpretation.md`
- `cost_sensitivity_interpretation.md`
- `imbalance_research_outputs/`
- `artifact_audit_outputs/`
- `future_work_outputs/`

## Files

- `job_postings_eda_model_comparison.ipynb`: local notebook with saved outputs.
- `job_postings_eda_model_comparison_colab.ipynb`: Colab-friendly notebook.
- `imbalance_focused_research.ipynb`: focused research notebook on class imbalance, metric choice, threshold sensitivity, and error patterns.
- `comprehensive_imbalance_experiments.py`: reusable experiment runner for prevalence, review budget, cost sensitivity, training balance, feature ablation, label scarcity, and holdout error profiling.
- `artifact_robustness_audit.py`: audit runner for duplicate leakage, split robustness, shortcut features, counterfactual credibility edits, subgroup robustness, and error case exports.
- `future_work_extensions.py`: near-duplicate/template leakage and fake-posting archetype discovery experiments.
- `related_work_and_research_gap.md`: literature-grounded research gap connecting prior future-work sections to this benchmark audit.
- `benchmark_audit_paper_outline.md`: paper-style outline with title, abstract draft, research questions, contributions, and reporting recommendations.
- `imbalance_research_plan.md`: project plan for the imbalance-focused continuation.
- `imbalance_research_summary.md`: written summary of the imbalance-focused findings.
- `comprehensive_imbalance_experiment_report.md`: detailed report interpreting the expanded imbalance experiments.
- `artifact_robustness_audit_report.md`: detailed report evaluating whether model performance is affected by artifacts, leakage, shortcuts, and robustness issues.
- `future_work_extension_report.md`: detailed report on near-duplicate/template leakage and fake-job archetype detection.
- `threshold_interpretation.md`: detailed explanation of threshold sensitivity plots and metrics.
- `cost_sensitivity_interpretation.md`: detailed explanation of cost sensitivity assumptions, selected thresholds, and tradeoffs.
- `project_framing.md`: research questions, hypotheses, methodology, and final project angle.
- `results_analysis.md`: written summary of the project results with key tables and plots.
- `Data/fake_jobs_cleaned.csv`: cleaned and feature-engineered dataset.
- `Data/fake_job_postings.csv`: original dataset.
- `results/`: generated charts and summary CSVs from the local run.
- `imbalance_research_outputs/`: generated tables and figures from the imbalance-focused notebook.

## Recommended Colab Workflow

1. Open `job_postings_eda_model_comparison_colab.ipynb` in Google Colab.
2. Run the setup cell.
3. If Colab cannot find the CSV files through Google Drive, upload:
   - `fake_jobs_cleaned.csv`
   - `fake_job_postings.csv`
4. Run the remaining cells.

The Colab notebook displays charts and tables inside the notebook and does not write new result files.

## Local Workflow

Install dependencies:

```bash
pip install -r requirements.txt
```

Open and run:

```bash
jupyter notebook job_postings_eda_model_comparison.ipynb
```

## Project Angle

The dataset is highly imbalanced, with fake postings making up about 4.84% of the rows. The project focuses on why accuracy is misleading for this task and compares classic machine learning models using imbalance-aware metrics such as average precision, ROC AUC, fake-class recall, and fake-class F1.

The extended notebook also includes weighted vs unweighted model comparisons, precision-recall curves, feature interpretation, and error analysis.

The newest audit reframes the project around trustworthiness: whether strong fake-job detection performance is stable when duplicate content, split strategy, shortcut features, and credibility metadata are tested directly.

The latest extension adds near-duplicate/template leakage analysis and unsupervised fake-job archetype discovery. This is the strongest research angle: EMSCAD performance appears sensitive to repeated templates, and aggregate fake-class recall hides which fake-posting archetypes are harder to detect.

The current working title is **Beyond Accuracy on EMSCAD: A Robustness and Shortcut Audit of Fake Job Posting Detection**.

