# Data Acquisition

---

## Overview

In this notebook, we establish the foundation for our machine learning project by acquiring and validating the datasets that will be used throughout the modeling pipeline. The data acquisition process is designed to ensure both reproducibility and extensibility, leveraging both structured and unstructured data sources.

---

## 1. Introduction

The project begins by collecting data from two complementary sources:

- **Structured Data:**We utilize the [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset, a well-known labeled dataset for customer churn prediction. This dataset provides a robust basis for initial model training and feature engineering, enabling us to explore patterns in customer behavior using structured, tabular data.
- **Unstructured Data:**
  To enrich our analysis, we also extract real-world customer comments from [Reddit](https://www.reddit.com/r/argentina/comments/1i924b2/movistar_av%C3%ADspense/) using the official Reddit API. This unstructured textual data allows us to incorporate sentiment analysis and capture nuanced customer perceptions that are not always reflected in traditional datasets.

---

## 2. Data Acquisition Workflow

### 2.1. Library Imports

We begin by importing all necessary libraries and utility functions for data loading and previewing. This ensures a modular and maintainable workflow.

### 2.2. Downloading and Loading the Telco Dataset

- The Telco Customer Churn dataset is automatically downloaded (if not already present) and stored in the `./data/raw/` directory.
- The dataset is loaded into a pandas DataFrame, and a data dictionary in markdown format is generated for documentation purposes.
- We verify the successful loading of the dataset by displaying its shape and a preview of the first 10 rows.
- Additional information about the dataset, such as column types and null values, is displayed to facilitate further analysis.

### 2.3. Acquiring Reddit Comments

- We configure and authenticate access to the Reddit API using a dedicated account.
- Reddit comments are fetched from a specified post and stored locally for reproducibility.
- The comments are loaded into a pandas DataFrame, and a preview is displayed to confirm successful acquisition.

---

## 3. Data Validation

- For both datasets, we perform initial validation by checking their shapes and previewing sample records.
- This step ensures that the data has been correctly loaded and is ready for further processing.

---

## 4. Next Steps

With both structured and unstructured data sources successfully acquired and validated, the next phase of the project will focus on **Exploratory Data Analysis (EDA)**. In the EDA notebook, we will:

- Explore the distributions, relationships, and potential issues within the Telco dataset.
- Analyze the sentiment and content of Reddit comments.
- Identify missing values, outliers, and feature engineering opportunities.
- Formulate hypotheses and guide the feature selection process for downstream modeling.

---

## Summary

This notebook has established a robust and reproducible data acquisition pipeline, integrating both tabular and textual data sources. By ensuring data quality and accessibility at this early stage, we set the stage for effective exploratory analysis and model development in subsequent notebooks.

---

**Proceed to the next notebook: _Exploratory Data Analysis (EDA)_**