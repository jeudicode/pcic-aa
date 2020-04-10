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

# seleccion de registros para entrenamiento y pruebas
data_train, data_test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=0)

mul = nb.MultinomialNB() # clasificador multinomial
ber = nb.BernoulliNB(binarize=1) # clasificador Bernoulli

# se configuran los datos
X = data_train.iloc[:,:2000]
y = data_train[[2000]]
X_test = data_test.iloc[:,:2000]
y_test = data_test[[2000]]

# aplicando el clasificador multinomial
pred_train = mul.fit(X, np.ravel(y)).predict(X)
pred_test = mul.fit(X, np.ravel(y)).predict(X_test)

# aplicando el clasificador Bernoulli
pred_train_ber = ber.fit(X, np.ravel(y)).predict(X)
pred_test_ber = ber.fit(X, np.ravel(y)).predict(X_test)

print("\n******* Resultados para el clasificador multinomial *******")
suma = (np.ravel(y) != pred_train).sum()
suma2 = (np.ravel(y_test) != pred_test).sum()
print("\nErrores de entre %d valores en el set de entrenamiento: %d" % (X.shape[0], suma))
print("Errores de entre %d valores en el set de prueba: %d" % (X_test.shape[0], suma2))
print("Precisión en el conjunto de entrenamiento: %d%%" % (100 - (suma / X.shape[0] * 100)))
print("Precisión en el conjunto de prueba: %d%%" % (100 - (suma2 / X_test.shape[0] * 100)))




print("\n******* Resultados para el clasificador Bernoulli *******")
suma = (np.ravel(y) != pred_train_ber).sum()
suma2 = (np.ravel(y_test) != pred_test_ber).sum()
print("\nErrores de entre %d valores en el set de entrenamiento: %d" % (X.shape[0], suma))
print("Errores de entre %d valores en el set de prueba: %d" % (X_test.shape[0], suma2))
print("Precisión en el conjunto de entrenamiento: %d%%" % (100 - (suma / X.shape[0] * 100)))
print("Precisión en el conjunto de prueba: %d%%" % (100 - (suma2 / X_test.shape[0] * 100)))
