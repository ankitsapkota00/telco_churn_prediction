# telco_churn_prediction
this is my first machine learning project
i predict which telecom customers will leave

the data : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

What i did : Cleaned the data ( including the blank TotalCharges values that isnull() couldn't see) encoded, categoricals, built a pipeline, trained a random forest
I used Claude to learn about evaluation metrics and threshold tuning.

Results

|version       |                accuracy    |       churnrecall    |      churnprecisoin |
|---|---|---|---|
|predict no on everyone   |         73.5      |           0        |             n/a|
|default random forest     |        ~77.5       |         45          |          59|
|+ stratify y and classweights |    ~78.3       |         46         |            63|
\+ threshold 0.3               |     75           |       73             |        53|

What i learned
Accuracy alone was misleading, because predicting "No" for everyone scores 73.5. Recall on the churn class is what matters. Recall means Of the people who really left, how many did I catch? A threshold of 0.3 caught far more churners at the cost of more false alarms.

limitations and next step
I chose threshold by looking at ehe same validation set so the numbers are slightly optimistic. Next steps: cross-validation, ROC-AUC, feature importances

