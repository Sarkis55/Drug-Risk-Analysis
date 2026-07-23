# 🏥 Drug Adverse Event Analysis and Risk Profiling

Predicting whether a drug adverse event will be clinically **serious**, using real-world data from the FDA's openFDA Adverse Event Reporting System (FAERS).

## Motivation

Regulatory agencies like the FDA receive millions of adverse event reports per year. Manually triaging which ones are serious is expensive and slow. This project asks: **can we predict whether a drug adverse event will be clinically serious — and which drugs, patient demographics, and reaction types are the strongest risk indicators?**

## Overview

This project analyses 800 real adverse event reports pulled live from the FDA's openFDA API across 8 common drugs, covering the full data science workflow: **ingestion → cleaning → EDA → visualization → feature engineering → hyperparameter tuning → ML modelling → evaluation.**

## Data Source

- **API**: [openFDA Drug Adverse Event API](https://api.fda.gov/drug/event.json) (public, no API key required)
- **Drugs queried**: Aspirin, Ibuprofen, Metformin, Lisinopril, Atorvastatin, Amoxicillin, Omeprazole, Metoprolol (100 records each)
- **Fields extracted**: drug name, patient age, patient sex, primary reaction, seriousness flag, reaction outcome

## Project Workflow

### 1. Data Extraction
Queries the openFDA API for each drug and parses out patient demographics, reactions, seriousness, and outcomes into a raw DataFrame.

### 2. Preprocessing
- Converts and validates data types
- Decodes FDA's numeric codes (sex, seriousness, outcome) into readable labels
- Filters invalid ages (keeps 0–120) and imputes missing ages with the per-drug median
- Produces a clean, analysis-ready dataset

### 3. Exploratory Data Analysis
Summarizes events and seriousness rate by drug, top reported reactions, sex breakdown, age statistics by seriousness, and outcome distribution.

### 4. Visualizations
- Adverse event reports by drug (serious vs. not serious)
- Top 15 reported adverse reactions
- Age distribution by seriousness
- Outcome distribution (pie chart)
- Feature importance (top 12)
- ROC curve

### 5. Feature Engineering
Builds a model-ready feature matrix via one-hot encoding of drug, sex, age group (0-17, 18-39, 40-59, 60-74, 75+), top 10 reaction categories, and outcome.

### 6. Hyperparameter Tuning
Uses `GridSearchCV` with stratified 5-fold cross-validation to tune a `GradientBoostingClassifier` over n_estimators, max_depth, learning_rate, and min_samples_leaf, and compares the tuned model against an untuned baseline.

### 7. Model Training & Evaluation
Trains a Gradient Boosting Classifier on the best hyperparameters and evaluates it on a held-out test set.

## Key Findings

**Adverse Event Landscape**
- Amoxicillin had the highest seriousness rate at 91%, followed by Atorvastatin (88%) and Ibuprofen (73%)
- Most commonly reported reactions: Type 2 Diabetes Mellitus, Gastrointestinal Haemorrhage, and Diarrhoea
- 68% of all events were classified as serious by reporters
- 42 reports resulted in death — all classified as serious

**Patient Demographics**
- Mean patient age: 61.3 years (serious: 59.9 | not serious: 65.0)
- Female patients slightly outnumbered male (441 vs 332), with similar seriousness rates (~68%)
- Most reports came from patients aged 53–73 (IQR)

**Model Performance**

| Metric | Score |
|---|---|
| Accuracy | 71.9% |
| ROC-AUC | 0.784 |
| Serious recall | 85% |
| Not-Serious recall | 43% |

- ROC-AUC of 0.78 shows meaningful discriminative power well above chance (0.5)
- Top predictors: drug type (Amoxicillin, Atorvastatin), fatal outcome, Type 2 Diabetes reaction
- The moderate accuracy reflects the real-world complexity of adverse event reporting — more data would likely improve results

## Tech Stack

- **Data**: `requests`, `pandas`, `numpy`
- **Visualization**: `matplotlib`
- **Modelling**: `scikit-learn` (`GradientBoostingClassifier`, `GridSearchCV`, `StratifiedKFold`)

## Getting Started

```bash
pip install requests pandas numpy matplotlib scikit-learn
jupyter notebook Drug_Adverse_Event_Analysis_and_Risk_Profiling.ipynb
```

Run all cells top to bottom — the notebook fetches fresh data from the openFDA API, so results may vary slightly between runs as the underlying dataset is updated.

## Why This Project

- **Real FDA data** — not a toy dataset; demonstrates working with public health APIs
- **End-to-end pipeline** — ingestion, cleaning, EDA, feature engineering, modelling, evaluation
- **Domain relevance** — pharmacovigilance (drug safety monitoring) is a core function in health tech, pharma, and insurance
- **Honest evaluation** — reports ROC-AUC alongside accuracy, and discusses class imbalance and its effect on results
