# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the
6. Required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Kailash S M
RegisterNumber: 21222040068
*/
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


# Read the CSV file
data = pd.read_csv("/content/Employee_EX6.csv")

# Display the first few rows of the data
print(data.head())

# Get information about the dataset
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Check the distribution of the target variable
print(data["left"].value_counts())

# Use LabelEncoder to encode the 'salary' column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Select features (X) and target variable (y)
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
y = data["left"]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(criterion="entropy")

# Train the classifier
dt.fit(x_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(x_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Predict on new data
prediction = dt.predict([[0.5, 0.8, 9, 260, 6, 0, 1, 2]])
print(prediction)

plt.figure(figsize=(18,6))
plot_tree(dt,feature_names=x.columns,class_names=['LEFT','NOT LEFT'],filled=True)
plt.show()


```

## Output:
![image](https://github.com/kailashmuthukumaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123893976/f2087e5c-d03a-4418-945c-298e47302107)
![image](https://github.com/kailashmuthukumaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123893976/fcbd8105-f7a0-44b7-a503-20efb99cdae6)



![image](https://github.com/kailashmuthukumaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123893976/235f97bd-6567-4809-8005-c326bf0ba719)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
