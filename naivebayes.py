import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix

df = pd.read_csv("/Users/aman/Desktop/Work/2020-21/dwm/datasets/mnist_train.csv")
df_test = pd.read_csv("/Users/aman/Desktop/Work/2020-21/dwm/datasets/mnist_test.csv")
# print(df.head())
# print(df.isnull().values.any())

X, y = df.iloc[:, 1:], df.iloc[:, 0]
X_test, y_test = df_test.iloc[:, 1:], df_test.iloc[:, 0]

bayes = GaussianNB()
model = bayes.fit(X, y)
y_pred = model.predict(X_test)
print("prediction for ten test cases:", model.predict(df_test.iloc[:10, 1:]))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))