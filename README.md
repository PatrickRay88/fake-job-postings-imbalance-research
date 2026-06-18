# Fake Job Postings ML Project

Exploratory data analysis and classic machine learning model comparison for an imbalanced fake job postings dataset.

## Separate Imbalance Research Focus

This repository is a separate continuation of the original class project. The added research focus is:

**How does class imbalance affect model evaluation, model selection, threshold choice, and error patterns in fake job posting detection?**

Start with:

- `imbalance_research_plan.md`
- `imbalance_focused_research.ipynb`
- `imbalance_research_summary.md`
- `threshold_interpretation.md`
- `cost_sensitivity_interpretation.md`
- `imbalance_research_outputs/`

## Files

- `job_postings_eda_model_comparison.ipynb`: local notebook with saved outputs.
- `job_postings_eda_model_comparison_colab.ipynb`: Colab-friendly notebook.
- `imbalance_focused_research.ipynb`: focused research notebook on class imbalance, metric choice, threshold sensitivity, and error patterns.
- `imbalance_research_plan.md`: project plan for the imbalance-focused continuation.
- `imbalance_research_summary.md`: written summary of the imbalance-focused findings.
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

