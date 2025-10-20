# Kaiburr Assessment - Task 5: Data Science (Text Classification)

This is a repository that holds the documentation and findings of Task 5. All analysis was done in Python via Google Colab and the original.ipynb notebook file is contained herein this repository.

This was to create a multi-class text classification model, which could group the consumer complaints into one of four product sets as stated in the assessment.

---

## Summary of Steps

The project was implemented in accordance with the 6 steps identified in the task instructions:

1. **Explanatory Data Analysis: The complaints.csv.zip file was directly loaded into a pandas DataFrame. The most important were identified including the columns- Product (the category) and Consumer complaint narrative (the text).
2. **Text Pre-Processing: the data was sanitized by eliminating rows that had empty complaints in them. It was subsequently filtered as to only the four target categories. A preprocess text function was developed to transform text to lower case and strip text of punctuations and digits.
3. **Feature Engineering: The text after cleaning was divided into an 80 training set and a 20 testing set. The text was converted to a numerical matrix of 10,000 features with a `TfidfVectorizer` and English stop words were automatically eliminated.
4. **Choosing Multi-Classification model: Two models have been chosen to be compared, namely: **Logistic Regression (with multi_class=ovr) and Multinomial Naive Bayes.
5. **Comparison and Model Assessment: Both models were trained using TF-IDF matrix. The reason for choosing the Logistic regression model of 90.9 percent is because it is the best and beats the Naive bayes model that has 87.7 percent accuracy.
6. Prediction: The categories of new, unseen strings of complaint were successfully predicted with the help of the trained Logistic Regression model.

---

## Colab Screenshot Proof

### 1. Data Loading & EDA

(Insert your "Data Loading" screenshot into here)

### 2. Data Filtering & Preparation

(Insert your "Data Filtration" screenshot in this box)

### 3. Model Evaluation (Step 5)

(Insert your screenshot of the "Model Evaluation" here)

### 4. Final Prediction (Step 6)

(Place your "Prediction" screenshot here)
