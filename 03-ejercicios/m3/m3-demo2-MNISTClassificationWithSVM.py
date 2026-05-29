# coding: utf-8

# # SVM Model for Image Classification
# ##### Using SVM to classify MNIST data - a set of images of hand-written digits

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

### MNIST data set: 
# Text images of 28x28 pixels represented as flattened array of 784 pixels
# Each pixel is represented by a pixel intensity value from 0-255
# Download Link: https://www.kaggle.com/c/3004/download/train.csv

print("Cargando el dataset MNIST...")
mnist_data = pd.read_csv("../data/mnist/train.csv")
print(mnist_data.tail())

#### Preparing our training and test data
# The pixel intensities are divided by 255 so that they're all between 0 and 1
features = mnist_data.columns[1:]
X = mnist_data[features]
Y = mnist_data['label']

X_train, X_test, Y_train, y_test = train_test_split(X/255., Y, test_size=0.1, random_state=0)

#### Create an SVM classifier model
print("\nEntrenando el modelo SVM inicial (L2)...")
clf_svm = LinearSVC(penalty="l2", dual=False, tol=1e-5)
clf_svm.fit(X_train, Y_train)

#### Calculate accuracy of the model against the test set
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f'SVM accuracy inicial: {acc_svm:.4f}')

### Grid Search
# - A brute-force way to obtain the best parameters for the ML algorithm
# - Tries out all combinations of parameters specified in the "grid"
# - Returns combination of parameters with the highest accuracy score
# - ATENCIÓN: Como explora todas las combinaciones, esto va a tardar bastante tiempo.

print("\nIniciando Grid Search (Por favor, ten paciencia, está probando combinaciones)...")
penalties = ['l1', 'l2']
tolerances = [1e-3, 1e-4, 1e-5]

param_grid = {'penalty': penalties, 'tol': tolerances}

# Especificamos loss='squared_hinge' para asegurar compatibilidad total con 'l1' y 'l2'
grid_search = GridSearchCV(LinearSVC(dual=False, loss='squared_hinge', random_state=0), param_grid, cv=3)
grid_search.fit(X_train, Y_train)

print("\n¡Grid Search completado!")
print("Mejores parámetros encontrados:", grid_search.best_params_)

#### Plugging in the "best parameters" to redefine the model 
# Usamos los mejores parámetros arrojados por el Grid Search (usualmente l1 con tol=1e-3)
best_penalty = grid_search.best_params_['penalty']
best_tol = grid_search.best_params_['tol']

print(f"\nEntrenando el modelo optimizado con penalty='{best_penalty}' y tol={best_tol}...")
clf_svm_optimized = LinearSVC(penalty=best_penalty, dual=False, tol=best_tol, loss='squared_hinge', random_state=0)
clf_svm_optimized.fit(X_train, Y_train)

# Evaluación final
y_pred_svm_opt = clf_svm_optimized.predict(X_test)
acc_svm_opt = accuracy_score(y_test, y_pred_svm_opt)
print(f'SVM accuracy optimizado: {acc_svm_opt:.4f}')