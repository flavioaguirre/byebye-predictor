# 📉 ByeBye Predictor - Customer Churn Prediction with Real Reviews

---

## 🧠 Introduction

This project delivers a machine learning solution designed to predict customer churn by analyzing both behavioral patterns and opinions expressed online.  

By first developing a robust baseline model using Kaggle’s **"Telco Customer Churn"** dataset, and later integrating customer sentiment insights from Reddit discussions, we achieve a hybrid predictive system that mimics real-world scenarios telecom companies face when trying to retain customers.  

This approach enables companies to **proactively identify at-risk customers and design targeted retention strategies**, ultimately improving customer lifetime value (CLV).  

---

## 📦 About the Datasets

1. **Structured dataset (customer profiles and churn labels):**  

   - **Name:** Telco Customer Churn  
   - **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
   - **Description:** Data from 7,043 customers, including demographics, contract details, payment history, and service usage.  
   - **Target variable:** `Churn` (Yes/No)  

2. **Unstructured dataset (customer opinions):**  

   - **Source:** [Reddit - Real User Reviews](https://www.reddit.com/r/argentina/comments/1i924b2/movistar_av%C3%ADspense/)  
   - **Acquisition:** Collected using the `PRAW API`.  
   - **Content:** Customer discussions about service quality, support experience, and billing issues.  

---

## 🧭 Project Flow

1. **Data Acquisition:**  
   - Download structured data from Kaggle.  
   - Retrieve customer opinions from Reddit using APIs.  

2. **Preprocessing and Cleaning:**  
   - Clean both the structured dataset and Reddit data.  
   - Normalize text (remove noise like URLs, emojis, symbols).  

3. **Baseline Modeling (Structured Data Only):**  
   - Conduct Exploratory Data Analysis (EDA).  
   - Apply feature engineering and perform feature selection to enrich the dataset.  
   - Train and evaluate multiple baseline models:  
     - `Logistic Regression`, `Random Forest`, `XGBoost`, `Decision Tree`, `KNN`.  
   - Optimize hyperparameters and compare models using metrics such as Accuracy, Precision, Recall, F1-score, and ROC AUC.  

4. **Integration of Unstructured Data:**  
   - Clean and process Reddit comments.  
   - Perform sentiment analysis using `TextBlob` and `VADER`.  
   - Identify key discussion topics with topic modeling (LDA).  

5. **Hybrid Feature Engineering:**  
   - Generate new features from text data:
     - `sentiment_score`: Sentiment polarity of customer opinions.  
     - `negative_comment_ratio`: Proportion of negative reviews.  
     - `dominant_topics`: Frequent discussion themes (e.g., "billing," "technical support").  
   - Simulate linking customer profiles to Reddit reviews to approximate real scenarios.  

6. **Hybrid Modeling and Evaluation:**  
   - Combine structured data and text-derived features into a unified dataset.  
   - Retrain models to assess the impact of unstructured data.  
   - Compare results with structured-only models to quantify performance gains.  

7. **Visualization and Insights:**  
   - Build interactive dashboards using Streamlit.  
   - Explore churn risk and simulate customer scenarios.  
   - Create visual reports showing how sentiment affects churn probability.  

---

## 📊 Key Results

- **Baseline Model (Structured Data Only):** 76% ROC AUC.  
- **Hybrid Model (Structured + Sentiment Features):** 87% ROC AUC.  
- **Impact:** +15% increase in recall for identifying churners after integrating Reddit sentiment data.  

---

## 📂 Project Structure

    📦 ByeByePredictor
    ├── assets/ # Static files (logos, images)
    ├── data/ # Structured datasets and opinions
    ├── models/ # Trained models
    ├── notebooks/ # Jupyter Notebooks with the full analysis
    ├── reports/ # Visualizations and documents
    ├── src/ # Project source code
    ├── README.md # This file
    └── requirements.txt # Environment dependencies

---

## 🛠️ Technologies Used

- 🐍 Python 3.x
- 📊 **Pandas**, **NumPy**, **Scikit-learn**
- 🧠 NLP: **TextBlob**, **VADER**, **gensim**, **nltk**, **spaCy**
- 📉 ML Algorithms: **XGBoost**, **Random Forest**, **Logistic Regression**, **Decision Tree**, **KNN**
- 🌐 APIs: **PRAW**
- 📚 Jupyter Notebooks
- 📈 Streamlit

---

## Installation

#### Step 1: Create a Virtual Environment

It is recommended that you create a virtual environment for this project. You can do this by following these steps:

```bash
python -m venv .venv
```

#### Step 2: Activate the virtual environment

On Linux/MacOS:

````bash
source .venv/bin/activate
````

On Windows:

````bash
.venv/Scripts/activate
````

#### Step 3: Install dependencies. Once the environment is activated, install all the necessary dependencies:

````bash
pip install -r requirements.txt
````

#### Step 4: Run the notebooks

Now you can start working with the notebooks. Go to the notebooks folder and explore the analyses.

---

## 📬 Contact

**Flavio Aguirre** – [LinkedIn](https://www.linkedin.com/in/flavio-aguirre-12784a252/) – [flavioaguirre0@gmail.com](mailto:flavioaguirre0@gmail.com)
