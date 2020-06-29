from sklearn import naive_bayes as nb
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# cargar datos

data_train = pd.read_csv("regl_data/juegos_entrenamiento.txt", sep=' ', header=None)

data_test = pd.read_csv("regl_data/juegos_validacion.txt", sep=' ', header=None)

X = data_train.iloc[:,:2]
y = data_train[[2]]
X_test = data_test.iloc[:,:2]
y_test = data_test[[2]]



# entrenando el clasificador bayesiano ingenuo

cat = nb.CategoricalNB()
estimator = cat.fit(X, np.ravel(y))
print("params: ", estimator.get_params())
pred_train = estimator.predict(X_test)
suma = (np.ravel(y_test) != pred_train).sum()

prec = (100 - (suma / X.shape[0]) * 100)

#new_data = cat.trans

print(suma, prec)
# np.random.seed(0)
# W = np.random.uniform(0, 1, size=(X.shape[1], 1))


# def sigmoid(z):
#     return (1 / 1 + np.exp(-z))

# def cross_entropy_loss(pred, target):
#     return -np.mean((target * np.log(pred) + (1 - target) * np.log(1 - pred)))

# def logistic_regression(X):
#     preds = []
#     for i in sigmoid(np.dot(X, W))