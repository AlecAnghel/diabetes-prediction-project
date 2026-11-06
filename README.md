# Diabetes Classification

A Python learning project that explores the Pima Indians Diabetes dataset and trains a random forest to classify the recorded diabetes outcome.

The notebook brings together data exploration, missing-value handling, model tuning, evaluation, and model export. It is an educational analysis, not a medical tool or a validated way to predict a person's future health.

## Start here

Open [the analysis notebook](ml_classifier_project/01_diabetes_classification.ipynb) to read the workflow, charts, and saved results.

## What the project includes

- Exploratory analysis of 768 records, eight input features, and the binary `Outcome` label.
- Checks for zero values in glucose, blood pressure, skin thickness, insulin, and BMI.
- A stratified 80/20 training and test split.
- A scikit-learn pipeline with median imputation, standard scaling, and a random forest.
- Five-fold grid search over tree count, maximum depth, and minimum split size.
- Classification reports, ROC-AUC scores, feature-importance plots, and a model export.

## Results and how to read them

The saved notebook outputs report:

| Model | Test accuracy | Test ROC-AUC |
| --- | ---: | ---: |
| Baseline random forest | 0.760 | 0.820 |
| Grid-search-selected model | 0.734 | 0.812 |

The selected model's best cross-validation ROC-AUC is 0.837. It performs worse than the baseline on this particular held-out split, which is a useful reminder that a better cross-validation score does not guarantee a better test score.

These values describe the current notebook. They should be read alongside the limitations below.

## Run locally

You will need Python 3, Git, and Jupyter Notebook.

```bash
git clone https://github.com/AlecAnghel/diabetes-prediction-project.git
cd diabetes-prediction-project
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib notebook
cd ml_classifier_project
jupyter notebook 01_diabetes_classification.ipynb
```

Run the cells from top to bottom. Start Jupyter from `ml_classifier_project` so the notebook can find `data/raw/diabetes[1].csv`. The final cell writes `diabetes_rf_pipeline.pkl` in that directory.

Package versions are not pinned in this repository, so results may vary between environments. Only load pickle or joblib files from sources you trust.

## Technical and design choices

- **Notebook-first workflow:** Explanations, code, plots, and outputs stay together so the analysis can be followed step by step.
- **Stratified splitting:** The split keeps the class proportions similar in the training and test sets.
- **Pipeline structure:** Preprocessing and the classifier are grouped into one object for fitting and export. Scaling is present in the current workflow, although random forests do not usually need it.
- **ROC-AUC for tuning:** Grid search compares the models' ranking of the two outcome classes rather than selecting on accuracy alone.
- **Feature-importance chart:** A compact view of all eight inputs helps explain what the fitted forest uses. It does not establish medical causes.

## Current limitations

- Zero values are replaced with full-dataset medians before the train/test split. This leaks information from the test set into preprocessing. A stronger evaluation would fit this step on training data only.
- The repeated-split section is unfinished: fitting and scoring sit outside the loop, so it collects one AUC score, not 30. The printed standard deviation is not a valid stability estimate.
- That section refits `best_pipeline` before export. The exported model is therefore not the same fitted instance evaluated in the final test section.
- The dataset is small and specific to its original population. There is no external validation or evidence of clinical usefulness.
- There is no automated test suite or pinned environment yet.

## Files

- `ml_classifier_project/01_diabetes_classification.ipynb`: main analysis.
- `ml_classifier_project/data/raw/diabetes[1].csv`: input dataset.
- `ml_classifier_project/diabetes_rf_pipeline.pkl`: saved model artifact.
