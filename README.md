<h1 align="center"><font size="7"><strong>📉 ByeBye Predictor - Customer Churn Prediction</strong></font></h1>

---

## 🧠 Introduction

In today’s highly competitive market, predicting customer churn is not just a technical challenge—it’s a **strategic advantage**.

This project was rebuilt from scratch to create a hybrid predictive system that combines structured data (contracts, payments, services) with unstructured opinions (forum comments, social media sentiment). The result is a robust, enterprise-ready model designed to **maximize recall**—capturing as many potential churners as possible—while keeping precision actionable.

This project showcases what happens when data science expertise and software engineering discipline work hand-in-hand:

- `Clean, modular code.`
- `Automated, reproducible pipelines.`
- `Insights that go beyond accuracy to generate real business value.`

This is not only a churn predictor but also the foundation for a framework that telecom companies can adopt to anticipate customer loss and design targeted retention strategies.

---

## 🎯 Project Objective

The goal is to predict customer churn by integrating **structured data** (customer demographics, contracts, billing) with **unstructured data** (real customer opinions from Reddit). By incorporating Natural Language Processing (NLP), we build models that not only estimate *who* is likely to leave but also uncover *why*, offering companies actionable insights for retention.

---

## 📦 About the Datasets

To capture churn risk from multiple perspectives, we use two complementary datasets:

**1. Structured Dataset — Customer Profiles and Churn Labels:**

- **Name:** Telco Customer Churn
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Scope:** 7,043 customer records with demographics, contract details, and service usage.
- **Business Value:** Provides the “hard facts” about how customers interact with the company, allowing us to model risk patterns grounded in contracts and payments.

**2. Unstructured Dataset — Customer Opinions and Complaints:**

- **Source:** [Reddit - Real User Reviews](https://www.reddit.com/r/argentina/comments/1i924b2/movistar_av%C3%ADspense/)
- **Acquisition:** Collected via the `PRAW API`, ensuring reproducibility.
- **Business Value:** Captures the emotional layer of the customer experience—the frustrations and praises that structured data cannot reflect.

---

## 🧭 Project Flow

Our workflow was designed to move from raw data to enterprise-ready models, ensuring reproducibility and business relevance at each step.

1. **Data Acquisition:**

   - Structured data is downloaded from Kaggle.
   - Unstructured data (Reddit comments) is extracted using the `PRAW` API.
   - Data is stored in `.csv` and `.json` for easy integration.
2. **Preprocessing & Cleaning:**

   - **Structured Data:** Handle missing values, encode categorical variables, and scale numerical features.
   - **Text Data:** Clean text by removing noise (URLs, emojis), tokenizing, and lemmatizing for NLP tasks.
3. **Feature Engineering:**

   - **From Text Data:** Generate features like `sentiment_score`, `negative_comment_ratio`, and `dominant_topics` (e.g., "billing," "support").
   - **From Structured Data:** Apply transformations to enrich customer attributes like contract type, monthly charges, and tenure.
4. **Modeling & Evaluation:**

   - **Models Tested:** Logistic Regression, Decision Tree, Random Forest, XGBoost, and more.
   - **Comparison:** A baseline model (structured data only) is compared against a hybrid model (structured + text features).
   - **Metrics:** Emphasis on **Recall**, the most critical metric for identifying at-risk customers, alongside Accuracy, Precision, and F1-score.
5. **Visualization & Reporting:**

   - Visuals are created to show feature importance, churn distributions, and the relationship between customer sentiment and churn probability.
   - All steps are documented in modular notebooks and reproducible pipelines.

---

## 📊 Key Results

- **Baseline Model (Structured Data Only):** **76%** ROC AUC.
- **Hybrid Model (Structured + Sentiment Features):** **87%** ROC AUC.
- **Impact:** **+15% increase in recall** for identifying churners after integrating Reddit sentiment data, proving the business value of unstructured data.

---

## 📂 Project Structure

```bash
📦 ByeByePredictor
├── assets/         		# Static files (logos, images)
├── data/           		# Raw and processed datasets
├── models/         		# Trained and serialized models
├── notebooks/      	   # Jupyter Notebooks for analysis and experimentation
├── reports/        		# Visualizations and generated documents
├── src/           	 	# Project source code (pipelines, utilities)
├── tests/          		# Unit tests for the source code
├── pyproject.toml  	   # Project configuration and dependencies
└── README.md      	   # This file
```

---

## 🛠️ Technologies Used

- **Core Language:** Python 3.x
- **Data Science & ML:** Pandas, NumPy, Scikit-learn, XGBoost
- **Natural Language Processing (NLP):** NLTK, spaCy, TextBlob
- **APIs & Data Acquisition:** PRAW
- **Testing & Reproducibility:** Pytest
- **Visualization:** Matplotlib, Seaborn
- **Development:** Jupyter Notebooks, VS Code

---

## Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/flavioaguirre/byebye-predictor.git
```

#### Step 2: Create a Virtual Environment

It is recommended that you create a virtual environment for this project. You can do this by following these steps:

```bash
python -m venv .venv
```

#### Step 3: Activate the virtual environment

On Linux/MacOS:

````bash
source .venv/bin/activate
````

On Windows:

````bash
.venv/Scripts/activate
````

#### Step 4: Install dependencies. Once the environment is activated, install all the necessary dependencies:

````bash
pip install -e .
````

#### Step 5: Run the notebooks

Now you can start working with the notebooks. Go to the notebooks folder and explore the analyses.

---

## Usage

Once installed, you can start exploring the project:

1. ``Run the analysis``: Navigate to the notebooks/ directory to see the step-by-step process, from data loading to modeling.
2. ``Run the pipelines``: Execute the Python scripts in the src/ directory to reproduce the data processing and model training pipelines.
3. ``Run tests``: Use Pytest to verify the functionality of the utility functions.


```bash
pytest -v
```

---

## 📬 Contact

**Flavio Aguirre** – [LinkedIn](https://www.linkedin.com/in/flavio-aguirre-12784a252/) – [flavioaguirre0@gmail.com](mailto:flavioaguirre0@gmail.com)
