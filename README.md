# 📉 ByeBye Predictor - Customer Churn Prediction with Real Reviews

---

## 🧠 Introduction

The goal of this project is to develop a machine learning model capable of predicting whether a customer will churn based on various characteristics related to their behavior and opinions expressed on forums or social media.

We will use Kaggle's renowned "Telco Customer Churn" dataset to train our model, as it has the initial labels necessary for a reasonably good baseline model. We will then simulate a more realistic environment by acquiring data from public comments about phone services on social media platforms like Reddit. This approach allows us to incorporate sentiment analysis and natural language-derived features as part of our predictive model.

---

## 📦 About the dataset

1. Structured dataset (customers and churn):

   * Name: Telco Customer Churn.
   * Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
   * Description: Dataset with information on 7,043 customers of a fictitious telecommunications company, including:Gender, contract type, services purchased Payment history, length of stay.
   * Target variable: Churn (Yes/No).


2. Unstructured data (customer comments)

   * Source: Reddit (subreddits such as r/movistar, r/claro, among others.)
   * Techniques: Extraction via Pushshift API and scraping with BeautifulSoup.
   * Content: Opinions on service quality, customer service, portability, and more.

---

## 🧭 Project Flow

1. **Data acquisition:**

   - Download the tabular dataset from Kaggle..
   - Extract real-world comments about telecommunications companies from Reddit.
   - Save in .csv and .json formats for easy integration.


2. **Text Cleaning and Processing:**

   - Remove text noise (URLs, emojis, symbols).
   - Lemmatization and stopword removal.
   - Sentiment analysis with ``TextBlob``.
   - Detection of frequent topics with LDA and word cloud visualization.


3. **Feature Engineering**

   - Generation of variables such as:
     * ``sentiment_score`` : puntaje de sentimiento del comentario asociado.
     * ``negative_comments``: number of negative reviews.
     * ``mentioned_topics``: topic tags (e.g., "billing," "technical support").
     * Synthetic association between each customer and a Reddit review based on profile characteristics, simulating a realistic environment.


4. **Target Labeling**

   - The original ``Churn`` variable from the Kaggle dataset is maintained as the target variable (``1``: churn / ``0``: maintained).
   - The relationship between negative sentiment and the likelihood of churn is investigated.


5. **Modeling and evaluation**

   - Modelos utilizados:

     * ``Logistic Regression``.
     * ``Random Forest``.
     * ``XGBoost``.
   - Comparison between trained models:

     * Only with structured data.
     * With structured data + sentiment analysis.
     * Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC AUC.


6. **Visualization and presentation**

   - Interactive Dashboard with Streamlit.
   - Customer Profile Exploration.
   - Real-time churn prediction.
   - Sentiment visualization and its impact on the decision.
   - Sentiment visualization and PDF report and Medium post explaining the approach and results and their impact on the decision.

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

    - 🐍 Python
    - 📊 Pandas, NumPy, Scikit-learn
    - 🧠 NLP: TextBlob, VADER, gensim, nltk, spaCy
    - 📉 XGBoost, Random Forest, Logistic Regression
    - 🌐 Web Scraping: Pushshift, BeautifulSoup
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
