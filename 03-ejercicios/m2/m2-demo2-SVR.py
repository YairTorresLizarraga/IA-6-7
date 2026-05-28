# coding: utf-8

# # Support Vector Regression
# ##### Using SVR to predict the MPG of vehicles

import pandas as pd
import numpy as np
import math
from pandas import Series
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

print(f"Versión de Pandas: {pd.__version__}")

### Download Auto MPG data set
# Link: https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data

# CORRECCIÓN 1: Se usa sep=r'\s+' en lugar de delim_whitespace=True
auto_data = pd.read_csv('../data/auto-mpg.data', sep=r'\s+', header=None,
                       names = ['mpg', 
                                'cylinders', 
                                'displacement', 
                                'horsepower', 
                                'weight', 
                                'acceleration',
                                'model', 
                                'origin', 
                                'car_name'])

print("\nPrimeras filas del dataset:")
print(auto_data.head())

#### Drop the car_name feature from the data frame
auto_data = auto_data.drop('car_name', axis=1)

#### Converting a numeric value for origin to something more meaningful
auto_data['origin'] = auto_data['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})

#### Applying one-hot-encoding
auto_data = pd.get_dummies(auto_data, columns=['origin'])

#### Convert missing values in data frame to NaN
auto_data = auto_data.replace('?', np.nan)

# DEBUG: Verificar si hay valores nulos por columna antes de borrarlos
print("\nValores nulos por columna antes de dropna():")
print(auto_data.isnull().sum())

#### Drop rows with missing values
auto_data = auto_data.dropna()

# DEBUG: Confirmar cuántas filas sobrevivieron la limpieza
print(f"\nFilas restantes después de dropna(): {len(auto_data)}")

### Prepare training and test data
X = auto_data.drop('mpg', axis=1)
Y = auto_data['mpg']

# División de datos limpia
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#### Define the Regression model
regression_model = SVR(kernel='linear', C=1.0)
regression_model.fit(X_train, Y_train)

#### Check the coefficients for each of the features
print("\nCoeficientes del modelo:")
print(regression_model.coef_)

#### Get R-square value with training data
train_score = regression_model.score(X_train, Y_train)
print(f"R-squared (entrenamiento): {train_score:.4f}")

#### Use matplotlib to view the coefficients as a histogram
# CORRECCIÓN 2: Se eliminó get_ipython().magic(u'matplotlib inline')
predictors = X_train.columns
coef = Series(regression_model.coef_[0], predictors).sort_values()

plt.figure() # Crea una ventana limpia para la gráfica
coef.plot(kind='bar', title='Modal Coefficients')
plt.tight_layout()
plt.show()

#### Get predictions on test data
y_predict = regression_model.predict(x_test)

#### Compare the predicted and actual values of the MPG
# CORRECCIÓN 3: Se eliminó get_ipython().magic(u'pylab inline') y se configuró el tamaño directamente en plt
plt.figure(figsize=(15, 6))

plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.ylabel('MPG')
plt.legend()
plt.show()

#### Get R-square value of predictions on test data
test_score = regression_model.score(x_test, y_test)
print(f"R-squared (prueba): {test_score:.4f}")

#### Calculate Mean Square Error
regression_model_mse = mean_squared_error(y_predict, y_test)
print(f"Mean Squared Error (MSE): {regression_model_mse:.4f}")

#### Root of Mean Square Error
rmse = math.sqrt(regression_model_mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")