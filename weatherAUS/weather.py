import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


datos = pd.read_csv('weatherAUS.csv')

datos.RainToday.replace(['Yes', 'No'], [1, 0], inplace=True)
datos.RainToday.value_counts()

datos.RainTomorrow.replace(['No', 'Yes'], [0, 1], inplace=True)
datos.RainTomorrow.value_counts()

datos.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm'], axis=1, inplace=True)

datos.dropna(axis=0, how='any', inplace=True)

train_data = datos[:38767]
test_data = datos[38767:]

x = np.array(train_data.drop(['RainTomorrow'], axis=1))
y = np.array(test_data.RainTomorrow)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

test_out_x = np.array(test_data.drop(['RainTomorrow'], axis=1))
test_out_y = np.array(test_data.RainTomorrow)



logreg = LogisticRegression(solver='lbfgs', max_iter=7600)


logreg.fit(train_x, train_y)


print('*'*50)

print('regresion Logistica')

print(f'accuracy de Test de Entrenamiento: {logreg.score(test_x, test_y)}')

print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {logreg.score(test_out_x, test_out_y)}')


svc = SVC(gamma='auto')

svc.fit(train_x, train_y)


print('*'*50)

print('maquina de soporte vectorial')

print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {svc.score(test_x, test_y)}')

print(f'accuracy de Validación: {svc.score(test_out_x, test_out_y)}')


arbol_random = RandomForestClassifier()

arbol_random.fit(train_x, train_y)


print('*'*50)

print('arbol random')

print(f'accuracy de Test de Entrenamiento: {arbol_random.score(test_x, test_y)}')

print(f'accuracy de Entrenamiento de Entrenamiento: {arbol_random.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {arbol_random.score(test_out_x, test_out_y)}')


arbol = DecisionTreeClassifier()

arbol.fit(train_x, train_y)

print('*'*50)

print('arbol de decision')

print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {arbol.score(test_x, test_y)}')

print(f'accuracy de Validación: {arbol.score(test_out_x, test_out_y)}')