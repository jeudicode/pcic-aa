from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# expansion polinomial
def expol(X, grado):
  phi_X = X
  for i in range(grado - 1):
    powerX = np.power(X, i + 2)
    phi_X = np.column_stack((phi_X, powerX))
    
  unos = np.ones(phi_X.shape[0])
  return np.column_stack((unos, phi_X))

# errores cuadraticos
def sse(y, y_hat):
  return np.square(y - y_hat).sum()

class RegresionLineal():
  def fit(self, X, y):
    '''
    Estima los parámetros del modelo
    '''
    self._estima_parametros(X, y)
  
  def predict(self, X):
    '''
    Predice valores de vectores dados
    '''
    return X @ self.parameters

class MinimosCuadrados(RegresionLineal):
  def _estima_parametros(self, X, y):
    self.parameters = np.linalg.inv(X.T @ X) @ (X.T @ y)

class MinimosCuadradosQR(RegresionLineal):
  def _estima_parametros(self, X, y):
    q, r = np.linalg.qr(X)
    self.parameters = np.linalg.inv(r) @ q.T @ y

def entrena_evalua(m, X_ent, y_ent, X_valid, y_valid, X_rango, grado):
  phi_X_ent = expol(X_ent, grado)
  phi_X_valid = expol(X_valid, grado)

  m.fit(phi_X_ent, y_ent)

  y_hat_ent = m.predict(phi_X_ent)
  y_hat_valid = m.predict(phi_X_valid)

  phi_X_rango = expol(X_rango, grado)
  y_hat_rango = m.predict(phi_X_rango)

  mse_ent = sse(y_ent, y_hat_ent) / n_ent
  mse_valid = sse(y_valid, y_hat_valid) / n_valid

  return y_hat_rango, mse_ent, mse_valid

# cargar datos

data = pd.read_csv("housing1.data", sep=',', header=None)

data_train, data_test = train_test_split(data, train_size=0.7, test_size=0.3, random_state=0)

# a) Mínimos cuadrados con expansion polinomial de diferentes grados

X = data_train.iloc[:,:13]
n_ent = X.shape[0]
y = data_train[[13]]
X_test = data_test.iloc[:,:13]
n_valid = X_test.shape[0]
y_test = data_test[[13]]
X_rango = np.linspace(0, 20, 10000)

parts = [1, 3, 5, 13] # particiones a probar
degrees = np.array([2, 5, 10]) # grados de polinomio a probar
mse_ent_grados = np.zeros(degrees.shape[0])
mse_valid_grados = np.zeros(degrees.shape[0])

for att in parts:
  deg_train = np.array([])
  deg_test = np.array([])
  for deg in degrees:
      for i in range(10): # 10 repeticiones
        shuffled_data = X.iloc[np.random.permutation(range(X.shape[0]))].copy()
        split = np.array_split(shuffled_data, 5) # 5 particiones
        fold_train = np.array([])
        fold_test = np.array([])

        for part in range(5):
          dtest = split[part]
          dtrain = shuffled_data.loc[~shuffled_data.index.isin(dtest.index)]
          
          ftrain = np.array(dtrain[:att])
          mtrain = expol(dtrain, y)


# b) Minimos cuadrados con expansion polinomial de grado 20 y penalizacion por norma l1 y l2 con diferentes valores de \lambda

# solo se utilizan 2 parámetros

# poly20 = PolynomialFeatures(degree=20)
# r = Ridge(alpha=50)
# lasso = Lasso(alpha=20,tol=100)
# lr = LinearRegression()

# x_lasso = data_train.iloc[:,:2]
# y_lasso = data_train[[13]]
# X_lasso_test = data_test.iloc[:,:2]
# y_lasso_test = data_test[[13]]

# x_lasso_poly = poly20.fit_transform(x_lasso)
# pred_linear = lr.fit(x_lasso_poly, np.ravel(y_lasso)).predict(poly20.transform(X_lasso_test))
# pred_lasso = lasso.fit(x_lasso_poly, np.ravel(y_lasso)).predict(poly20.transform(X_lasso_test))


# x_ridge_poly = poly20.fit_transform(x_lasso)
# pred_linear = lr.fit(x_ridge_poly, np.ravel(y_lasso)).predict(poly20.transform(X_lasso_test))
# pred_ridge = r.fit(x_ridge_poly, np.ravel(y_lasso)).predict(poly20.transform(X_lasso_test))

# print("**** Lasso *****")
# print('Mean squared error: %.2f'
#       % mean_squared_error(y_lasso_test, pred_lasso))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(y_lasso_test, pred_lasso))

# print("**** Ridge *****")
# print('Mean squared error: %.2f'
#       % mean_squared_error(y_lasso_test, pred_ridge))
# # The coefficient of determination: 1 is perfect prediction
# print('Coefficient of determination: %.2f'
#       % r2_score(y_lasso_test, pred_ridge))



# c) Minimos cuadrados con expansion polinomial de grado 2 y seleccion de atributos

X = data_train.iloc[:,:13]
y = data_train[[13]]
X_test = data_test.iloc[:,:13]
y_test = data_test[[13]]

X_new = SelectKBest(f_regression, k=4).fit_transform(X, np.ravel(y))
X_test_new = SelectKBest(f_regression, k=4).fit_transform(X_test, np.ravel(y_test))

poly_reg = PolynomialFeatures(degree=2)

X_poly = poly_reg.fit_transform(X_new)
lr  = LinearRegression()
pred = lr.fit(X_poly, np.ravel(y)).predict(poly_reg.transform(X_test_new))


print("**** AAAAA *****")
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, pred))


# Plot outputs
# plt.scatter(X_test, y_test,  color='black')
# plt.plot(X_test, y_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())

# plt.show()
