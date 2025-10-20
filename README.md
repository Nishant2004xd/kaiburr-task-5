# Kaiburr Assessment - Task 5: Data Science (Text Classification)

This repository contains the documentation and console output for Task 5. The entire analysis was performed in Python using Google Colab. The goal was to build a multi-class text classification model to categorize consumer complaints into one of four product categories.

---

## Project Workflow & Output

Here is the console output for each of the 6 required steps.

### 1. Explanatory Data Analysis and Feature Engineering

**Explanatory Data Analysis (EDA):**

```

First, the data was loaded, analyzed, and engineered into numerical features.

Loading dataset... this may take a minute.
Dataset loaded successfully!
  Date received                                            Product  \
0    2020-07-06  Credit reporting, credit repair services, or o...   
1    2025-10-14  Credit reporting or other personal consumer re...   
2    2025-10-10  Credit reporting or other personal consumer re...   
3    2025-10-15  Credit reporting or other personal consumer re...   
4    2025-10-17  Credit reporting or other personal consumer re...   

        Sub-product                                 Issue  \
0  Credit reporting  Incorrect information on your report   
1  Credit reporting  Incorrect information on your report   
2  Credit reporting  Incorrect information on your report   
3  Credit reporting  Incorrect information on your report   
4  Credit reporting  Incorrect information on your report   

                                           Sub-issue  \
0                Information belongs to someone else   
1  Information is missing that should be on the r...   
2                Information belongs to someone else   
3                Information belongs to someone else   
4                Information belongs to someone else   

  Consumer complaint narrative  \
0                          NaN   
1                          NaN   
2                          NaN   
3                          NaN   
4                          NaN   

                             Company public response  \
0  Company has responded to the consumer and the ...   
1                                                NaN   
2                                                NaN   
3                                                NaN   
4                                                NaN   

                                  Company State ZIP code Tags  \
0     Experian Information Solutions Inc.    FL    346XX  NaN   
1                           EQUIFAX, INC.    TX    75062  NaN   
2                           EQUIFAX, INC.    GA    30341  NaN   
3  TRANSUNION INTERMEDIATE HOLDINGS, INC.    TX    75287  NaN   
4  TRANSUNION INTERMEDIATE HOLDINGS, INC.    NC    27127  NaN   

  Consumer consent provided? Submitted via Date sent to company  \
0                      Other           Web           2020-07-06   
1                        NaN           Web           2025-10-14   
2                        NaN           Web           2025-10-10   
3                        NaN           Web           2025-10-15   
4                        NaN           Web           2025-10-17   

  Company response to consumer Timely response? Consumer disputed?  \
0      Closed with explanation              Yes                NaN   
1                  In progress              Yes                NaN   
2                  In progress              Yes                NaN   
3                  In progress              Yes                NaN   
4                  In progress              Yes                NaN   

   Complaint ID  
0       3730948  
1      16558024  
2      16507707  
3      16593757  
4      16649455  

Column Names:
Index(['Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue',
       'Consumer complaint narrative', 'Company public response', 'Company',
       'State', 'ZIP code', 'Tags', 'Consumer consent provided?',
       'Submitted via', 'Date sent to company', 'Company response to consumer',
       'Timely response?', 'Consumer disputed?', 'Complaint ID'],
      dtype='object')

```
**Feature Engineering (TF-IDF):**
```
Original data shape: (11535877, 18)
Shape after dropping empty complaints: (3416745, 18)
Shape after filtering for 4 categories: (1323496, 18)

Data is ready for processing.
Total samples: 1323496

Category to Number Mapping:
'Consumer Loan': 0
'Credit reporting, credit repair services, or other personal consumer reports': 1
'Debt collection': 2
'Mortgage': 3

```

### 2. Text Pre-Processing

The raw data was cleaned, filtered, and the labels were encoded.
```
Starting text pre-processing and data split...
Text pre-processing complete.
Training samples: 1058796
Testing samples: 264700

Starting TF-IDF vectorization (Feature Engineering)...
Feature engineering complete.
Shape of TF-IDF matrix for training data: (1058796, 10000)

```

### 3. Selection of Multi Classification model

Two models, **Logistic Regression** and **Multinomial Naive Bayes**, were selected and trained.

'''
--- Step 4: Model Selection & Training ---

Training Logistic Regression model...
/usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1256: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. Use OneVsRestClassifier(LogisticRegression(..)) instead. Leave it to its default value to avoid this warning.
  warnings.warn(
Logistic Regression training complete.

Training Naive Bayes model...
Naive Bayes training complete.

'''

### 4. Comparison of model performance

The two trained models were compared on the test dataset. **Logistic Regression** (90.9% accuracy) performed better than Naive Bayes (87.7% accuracy).

```
--- Step 5: Model Comparison & Evaluation ---

--- Logistic Regression Evaluation ---
Accuracy: 0.9091

Classification Report:
                                                                              precision    recall  f1-score   support

                                                               Consumer Loan       0.73      0.30      0.42      1892
Credit reporting, credit repair services, or other personal consumer reports       0.92      0.95      0.93    161456
                                                             Debt collection       0.88      0.84      0.86     74380
                                                                    Mortgage       0.92      0.93      0.93     26972

                                                                    accuracy                           0.91    264700
                                                                   macro avg       0.86      0.75      0.78    264700
                                                                weighted avg       0.91      0.91      0.91    264700


--- Naive Bayes Evaluation ---
Accuracy: 0.8769

Classification Report:
                                                                              precision    recall  f1-score   support

                                                               Consumer Loan       0.45      0.27      0.34      1892
Credit reporting, credit repair services, or other personal consumer reports       0.90      0.92      0.91    161456
                                                             Debt collection       0.85      0.77      0.81     74380
                                                                    Mortgage       0.82      0.95      0.88     26972

                                                                    accuracy                           0.88    264700
                                                                   macro avg       0.75      0.73      0.73    264700
                                                                weighted avg       0.88      0.88      0.87    264700


```

### 5. Model Evaluation

The detailed **Classification Reports** above serve as the evaluation for both models. Based on the higher accuracy and weighted F1-score, the **Logistic Regression** model was selected as the final, superior model for this task.

### 6. Prediction

The trained Logistic Regression model was used to predict the categories of four new, unseen sample complaints.

```
--- Step 6: Prediction ---
Predicting categories for 4 new complaints...

Complaint: "I checked my credit report and there is an account that does not belong to me!"
--> Predicted Category: Credit reporting, credit repair services, or other personal consumer reports

Complaint: "A company keeps calling my cell phone trying to collect a debt that I already paid off."
--> Predicted Category: Debt collection

Complaint: "My application for a car loan was denied, and I don't know why."
--> Predicted Category: Credit reporting, credit repair services, or other personal consumer reports

Complaint: "I am having an issue with my mortgage escrow account, the payment is wrong."
--> Predicted Category: Mortgage

```
