from sklearn import naive_bayes as nb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# cargar datos

data = pd.read_csv("spam.csv", sep=' ', header=None)

# total de registros
total = len(data)

# diferencia entre spam y no spam
total_spam = data[data[2000] == 1][0].count()
total_no_spam = data[data[2000] == 0][0].count()
pct_spam = (total_spam / total) * 100
pct_no_spam = (total_no_spam / total) * 100
print("Total spam: ", total_spam)
print("Total no spam: ", total_no_spam)
print("Pct spam: ", round(pct_spam))
print("Pct no spam: ", round(pct_no_spam))

# selección de registros para entrenamiento y pruebas
data_train, data_test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=0)

print("shape: ", data_train.shape)

index = data_train.index
cols = data_train.columns
values = data_train.values

print(index)
print(cols)
print(values)
mul = nb.MultinomialNB()

X = data_train.iloc[:,:2000]
y = data_train[[2000]]
X_test = data_test.iloc[:,:2000]
y_test = data_test[[2000]]

pred_mul = mul.fit(X, np.ravel(y)).predict(X_test) #mul.fit(data_train).predict(data_test)

print("\nErrores de entre %d valores: %d" % (X_test.shape[0], (np.ravel(y_test) != pred_mul).sum()))
