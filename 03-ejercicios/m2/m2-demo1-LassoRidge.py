# -*- coding: utf-8 -*-

# # Lasso and Ridge Regression
# ##### First use Linear Regression to predict automobile prices. Then apply Lasso and Ridge Regression models on the same data and compare results

import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib as pylab
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

print(f"Pandas version: {pd.__version__}")

# ### Download the Automobile data set
data_path = 'imports-85.data' if os.path.exists('imports-85.data') else '../data/imports-85.data'

# Se lee el archivo limpiando espacios alrededor de las comas
auto_data = pd.read_csv(data_path, sep=r'\s*,\s*', engine='python')

# #### Fill missing values with NaN
auto_data = auto_data.replace('?', np.nan)

# #### Data Cleaning
# Convert the values in the price column to numeric values
auto_data['price'] = pd.to_numeric(auto_data['price'], errors='coerce') 

# Dropping a column which we deem unnecessary
auto_data = auto_data.drop('normalized-losses', axis=1)

# Horsepower is also non-numeric... so this is also converted to a numeric value
auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce') 

# Mapeo explícito para el número de cilindros
cylinders_dict = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}
auto_data['num-of-cylinders'] = auto_data['num-of-cylinders'].replace(cylinders_dict)

# #### One-Hot-Encoding 
auto_data = pd.get_dummies(auto_data, 
                           columns=['make', 
                                    'fuel-type', 
                                    'aspiration', 
                                    'num-of-doors', 
                                    'body-style', 
                                    'drive-wheels', 
                                    'engine-location', 
                                    'engine-type', 
                                    'fuel-system'],
                           dtype=int) 

# Drop rows containing missing values
auto_data = auto_data.dropna()
for col in auto_data.columns:
    if col != 'price':
        auto_data[col] = auto_data[col].astype(float)
# ### Data Cleaning is now complete. Build models.
X = auto_data.drop('price', axis=1)
Y = auto_data['price']

# Spliting into 80% for training set and 20% for testing set
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Configuración de gráficos
pylab.rcParams['figure.figsize'] = (15, 6)


# ==========================================
# 1. LINEAR REGRESSION
# ==========================================
print("\n--- Linear Regression ---")
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

print(f"R-square en entrenamiento: {linear_model.score(X_train, Y_train)}")

predictors = X_train.columns
coef = pd.Series(linear_model.coef_, predictors).sort_values()
print("Coeficientes ordenados:")
print(coef.head()) # Muestra los primeros para no saturar la consola

y_predict = linear_model.predict(x_test)

# Graficar
plt.figure()
plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Price')
plt.title('Linear Regression - Actual vs Predicted')
plt.legend()
plt.show()

linear_model_mse = mean_squared_error(y_predict, y_test)
print(f"R-square en test: {linear_model.score(x_test, y_test)}")
print(f"RMSE: {math.sqrt(linear_model_mse)}")


# ==========================================
# 2. LASSO REGRESSION
# ==========================================
print("\n--- Lasso Regression ---")
# SOLUCIÓN: Usamos un pipeline con StandardScaler para reemplazar `normalize=True`
lasso_model = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.5, max_iter=10000))
lasso_model.fit(X_train, Y_train)

print(f"R-square en entrenamiento: {lasso_model.score(X_train, Y_train)}")

y_predict = lasso_model.predict(x_test)

# Graficar
plt.figure()
plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Price')
plt.title('Lasso Regression - Actual vs Predicted')
plt.legend()
plt.show()

lasso_model_mse = mean_squared_error(y_predict, y_test)
print(f"R-square en test: {lasso_model.score(x_test, y_test)}")
print(f"RMSE: {math.sqrt(lasso_model_mse)}")


# ==========================================
# 3. RIDGE REGRESSION
# ==========================================
print("\n--- Ridge Regression ---")
# SOLUCIÓN: Usamos un pipeline con StandardScaler para reemplazar `normalize=True`
ridge_model = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=0.05))
ridge_model.fit(X_train, Y_train)

print(f"R-square en entrenamiento: {ridge_model.score(X_train, Y_train)}")

y_predict = ridge_model.predict(x_test)

# Graficar
plt.figure()
plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Price')
plt.title('Ridge Regression - Actual vs Predicted')
plt.legend()
plt.show()

ridge_model_mse = mean_squared_error(y_predict, y_test)
print(f"R-square en test: {ridge_model.score(x_test, y_test)}")
print(f"RMSE: {math.sqrt(ridge_model_mse)}")


# ==========================================
# 4. SUPPORT VECTOR REGRESSION (SVR)
# ==========================================
print("\n--- Support Vector Regression (SVR) ---")
regression_model = SVR(kernel='linear', C=1.0)
regression_model.fit(X_train, Y_train)

print(f"R-square en entrenamiento: {regression_model.score(X_train, Y_train)}")

y_predict = regression_model.predict(x_test)

# Graficar
plt.figure()
plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('Price')
plt.title('SVR - Actual vs Predicted')
plt.legend()
plt.show()

regression_model_mse = mean_squared_error(y_predict, y_test)
print(f"R-square en test: {regression_model.score(x_test, y_test)}")
print(f"RMSE: {math.sqrt(regression_model_mse)}")