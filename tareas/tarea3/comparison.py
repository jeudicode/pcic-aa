from sklearn import naive_bayes as nb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


# cargar datos

data = pd.read_csv("breast-cancer-wisconsin.data", sep=',', header=None)

# total de registros
total = len(data)

# Analizando el conjunto de datos, se encuentran datos faltantes para el atributo 6
# entonces se harán tres conjuntos rellenando la media, mediana y moda de este atributo
# respectivamente.

# Se obtienen los registros que están completos y se obtiene la media:
complete = data[data[6] != "?"].copy()
complete[6] = complete[6].astype("int64")

mean = complete[6].mean()
median = complete[6].median()
mode = complete[6].mode()[0]

# Se crea nuevos conjuntos con los valores completos

mean_set = data.copy()
median_set = data.copy()
mode_set = data.copy()
mean_set = mean_set.replace("?", mean)
mode_set = mode_set.replace("?", mode)
median_set = median_set.replace("?", median)

mean_set[10] = mean_set[10].astype("int64")
mode_set[10] = mode_set[10].astype("int64")
median_set[10] = median_set[10].astype("int64")

# Dado que no se utilizará, se elimina la columna ID

mean_set = mean_set.drop(columns=0)
median_set = median_set.drop(columns=0)
mode_set = mode_set.drop(columns=0)

print(mean_set.iloc[23])
print(median_set.iloc[23])
print(mode_set.iloc[23])

reps = 10
folds = 5

# selección de registros para entrenamiento y pruebas
mean_train, mean_test = train_test_split(mean_set, train_size=0.7, test_size=0.3, random_state=0)
median_train, median_test = train_test_split(median_set, train_size=0.7, test_size=0.3, random_state=0)
mode_train, mode_test = train_test_split(mode_set, train_size=0.7, test_size=0.3, random_state=0)

#
g = nb.GaussianNB() # clasificador de distribución normal

 # se configuran los datos
X_mean = mean_train.iloc[:,:10]
y_mean = mean_train[[10]]
X_mean_test = mean_test.iloc[:,:10]
y_mean_test = mean_test[[10]]

y_mean = np.ravel(y_mean)
y_mean_test = np.ravel(y_mean_test)


res_bay_rep_train = []
res_bay_rep_test = []


for rep in range(reps):
    shuff = X_mean.iloc[np.random.permutation(range(X_mean[0]))].copy()

    # separando por clase
    malign = shuff[shuff[10] == 2]
    benign = shuff[shuff[10] == 4]

    # creando la particion
    split_malign = np.array_split(malign, folds)
    split_benign = np.array_split(benign, folds)

    res_bay_fold_test = []
    res_bay_fold_train = []

    for fold in range(folds):
        malign_test = split_malign[fold]
        malign_train = split_malign.loc[~benign.index.isin()]


# aplicando el clasificador normal
pred_mean_train = g.fit(X_mean, y_mean).predict(X_mean)
pred_mean_test = g.fit(X_mean, y_mean).predict(X_mean_test)
pred_median_train = g.fit(X_median, y_median).predict(X_median)
pred_median_test = g.fit(X_median, y_median).predict(X_median_test)
pred_mode_train = g.fit(X_mode, y_mode).predict(X_mode)
pred_mode_test = g.fit(X_mode, y_mode).predict(X_mode_test)

print("\n******* Resultados para conjunto media *******")
suma = (y_mean != pred_mean_train).sum()
suma2 = (y_mean_test != pred_mean_test).sum()
print("\nErrores de entre %d valores en el set de entrenamiento: %d" % (X_mean.shape[0], suma))
print("Errores de entre %d valores en el set de prueba: %d" % (X_mean_test.shape[0], suma2))
print("Precisión en el conjunto de entrenamiento: %d%%" % (100 - (suma / X_mean.shape[0] * 100)))
print("Precisión en el conjunto de prueba: %d%%" % (100 - (suma2 / X_mean_test.shape[0] * 100)))

print("\n******* Resultados para conjunto mediana *******")
suma = (y_median != pred_median_train).sum()
suma2 = (y_median_test != pred_median_test).sum()
print("\nErrores de entre %d valores en el set de entrenamiento: %d" % (X_median.shape[0], suma))
print("Errores de entre %d valores en el set de prueba: %d" % (X_median_test.shape[0], suma2))
print("Precisión en el conjunto de entrenamiento: %d%%" % (100 - (suma / X_median.shape[0] * 100)))
print("Precisión en el conjunto de prueba: %d%%" % (100 - (suma2 / X_median_test.shape[0] * 100)))

print("\n******* Resultados para conjunto moda *******")
suma = (y_mode != pred_mode_train).sum()
suma2 = (y_mode_test != pred_mode_test).sum()
print("\nErrores de entre %d valores en el set de entrenamiento: %d" % (X_mode.shape[0], suma))
print("Errores de entre %d valores en el set de prueba: %d" % (X_mode_test.shape[0], suma2))
print("Precisión en el conjunto de entrenamiento: %d%%" % (100 - (suma / X_mode.shape[0] * 100)))
print("Precisión en el conjunto de prueba: %d%%" % (100 - (suma2 / X_mode_test.shape[0] * 100)))
