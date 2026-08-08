# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the placement dataset, remove unnecessary columns, check for missing/duplicate values, and encode categorical features using Label Encoding.

2.Split the processed data into input features (X) and target variable (status), then divide it into training and testing datasets.

3.Train a Logistic Regression model using the training data and predict placement status for the test data as well as for a new student.

4.Evaluate the model using Accuracy and Classification Report (Precision, Recall, and F1-score) to measure prediction performance.

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Krithika Lakshmi M
RegisterNumber: 212224230134
```
```py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Placement_Data.csv")

print("First 5 rows of the dataset:")
print(data.head())


data1 = data.copy()

data1 = data1.drop(["sl_no", "salary"], axis=1)

print("\nData after dropping 'sl_no' and 'salary':")
print(data1.head())


print("\nChecking for missing values (True = missing):")
print(data1.isnull().any())

print("\nNumber of duplicate rows:")
print(data1.duplicated().sum())

cat_cols = ["gender", "ssc_b", "hsc_b", "hsc_s", 
            "degree_t", "workex", "specialisation", "status"]

le = LabelEncoder()

for col in cat_cols:
    data1[col] = le.fit_transform(data1[col])

print("\nData after Label Encoding:")
print(data1.head())

X = data1.iloc[:, :-1]
y = data1["status"]

print("\nFeatures (X) sample:")
print(X.head())

print("\nTarget (y) sample:")
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

print("\nTraining and testing shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

lr = LogisticRegression(solver="liblinear")

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("\nPredicted values (y_pred):")
print(y_pred)


accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_student = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]

new_prediction = lr.predict(new_student)

print("\nPrediction for new student (0 = Not Placed, 1 = Placed):")
print(new_prediction[0])



```

## Output:

<img width="737" height="315" alt="image" src="https://github.com/user-attachments/assets/c56be76c-829d-4577-8cea-c2873b7aeadf" />

<img width="738" height="321" alt="image" src="https://github.com/user-attachments/assets/370a4e62-fdbf-468f-9de6-71b0103d24a5" />

<img width="482" height="407" alt="image" src="https://github.com/user-attachments/assets/bb7be0de-6025-467b-8947-55548ae72c9f" />

<img width="747" height="331" alt="image" src="https://github.com/user-attachments/assets/10a7bc86-26d9-4732-9307-d0e0862d8e9a" />

<img width="747" height="498" alt="image" src="https://github.com/user-attachments/assets/2b626824-c2c6-45c4-9715-d43f5c9b3d3a" />

<img width="403" height="92" alt="image" src="https://github.com/user-attachments/assets/62707a37-6423-4aff-b656-8e380300a5e3" />

<img width="735" height="92" alt="image" src="https://github.com/user-attachments/assets/66ec389f-b40b-45a3-9cf6-26bceec88c31" />

<img width="552" height="270" alt="image" src="https://github.com/user-attachments/assets/b5febec0-dfa2-4bfc-ba89-8c7ce3c5ee6b" />

<img width="552" height="52" alt="image" src="https://github.com/user-attachments/assets/60b16fc3-01f4-4900-9482-cb22dad3577a" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
