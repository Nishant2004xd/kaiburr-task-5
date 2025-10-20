# Kaiburr Assessment - Task 5: Data Science (Text Classification)

This repository contains the documentation and console output for Task 5. The entire analysis was performed in Python using Google Colab. The goal was to build a multi-class text classification model to categorize consumer complaints into one of four product categories.

---

## Project Workflow & Output

Here is the console output for each of the 6 required steps.

### 1. Explanatory Data Analysis and Feature Engineering

First, the data was loaded, analyzed, and engineered into numerical features.
'''
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

'''

**Explanatory Data Analysis (EDA):**
