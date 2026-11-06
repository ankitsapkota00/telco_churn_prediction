import pandas as pd
import numpy as np

data = pd.read_csv("Telco-Customer-Churn.csv", index_col = "customerID") # so it doesnt take customer id and to find stuff for analysis

for col in data.columns:
    if data[col].dtype == "object":                  # Find text columns
        data[col] = data[col].str.strip()            # Strip outer spaces
        data[col] = data[col].replace("", np.nan)    # Turn pure empty spaces into real NaN

print(data.isnull().sum()) #now total charges has 11


#theres categorical values so lets deal with it first 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, random_state = 0, class_weight='balanced')

numerical_column =['TotalCharges']
categorical_columns = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

#making pipleline although not needed for this one
categorical_transformer = Pipeline(steps=[('cat', OneHotEncoder(handle_unknown='ignore'))]) 
numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0))])

#column transformer
preprocessor = ColumnTransformer(transformers=[('imputer', numerical_transformer,numerical_column),
    ('ohe', categorical_transformer, categorical_columns)], remainder='passthrough') #remainder passthrough so it doesnt drop my numerical columns

my_pipeline = Pipeline(steps=[('col', preprocessor),
                              ('model', model)])


# if no missing values then i guess i can import skicit learn, oh first i need to split data into target and train
X = data.drop(["Churn"], axis = 1)
y = data.Churn

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0, stratify=y)


my_pipeline.fit(X_train,y_train)
preds = my_pipeline.predict(X_valid)


#lets see the models accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_valid, preds)
print(accuracy * 100)




# i used to think empty values are NaN but this data had blank
# spaces which made it undectable with print(data.isnull().sum())
# bunch of these code could be done in less lines but for now im keeping it very simple

# The evaluation and threshold sections below were written with help from
# Claude (an AI assistant). I used them to learn about confusion matrices,
# precision/recall, and threshold tuning, and I can explain each line.

from sklearn.metrics import confusion_matrix, classification_report

# --- Evaluate with the default 0.5 cutoff ---
preds = my_pipeline.predict(X_valid)
print("Accuracy:", accuracy_score(y_valid, preds))
print(confusion_matrix(y_valid, preds))
print(classification_report(y_valid, preds))

# --- Evaluate with a lower threshold to catch more churners ---
THRESHOLD = 0.3  # chosen from a threshold sweep; see README
proba = my_pipeline.predict_proba(X_valid)[:, 1]  # probability of "Yes"
print(proba[:5])
preds_tuned = np.where(proba >= THRESHOLD, "Yes", "No")
print(f"--- Threshold {THRESHOLD} ---")
print(confusion_matrix(y_valid, preds_tuned))
print(classification_report(y_valid, preds_tuned))