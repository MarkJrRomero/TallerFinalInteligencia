import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


simplefilter(action='ignore', category=FutureWarning)

datos = pd.read_csv('bank-full.csv')

datos.marital.replace(['married', 'single', 'divorced'], [2, 1, 0], inplace= True)
datos.education.replace(['unknown', 'primary', 'secondary', 'tertiary'], [0, 1, 2, 3], inplace= True)
datos.default.replace(['no', 'yes'], [0, 1], inplace= True)
datos.housing.replace(['no', 'yes'], [0, 1], inplace= True)
datos.loan.replace(['no', 'yes'], [0, 1], inplace= True)
datos.y.replace(['no', 'yes'], [0, 1], inplace= True)
datos.contact.replace(['unknown', 'cellular', 'telephone'], [0, 1, 2], inplace= True)
datos.poutcome.replace(['unknown', 'failure', 'other', 'success'], [0, 1, 2, 3], inplace= True)

datos.drop(['balance', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'job'], axis=1, inplace=True)
datos.age.replace(np.nan, 41, inplace=True)

rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']

datos.age = pd.cut(datos.age, rangos, labels=nombres)
datos.dropna(axis=0, how='any', inplace=True)


train_data = datos[:22605]
test_data = datos[22605:]

x = np.array(train_data.drop(['y'], 1))
y = np.array(train_data.y)

train_x, test_x, train_y, test_y = train_test_split(x, y, size_test=0.2)

test_out_x = np.array(test_data.drop(['y'], 1))
test_out_y = np.array(test_data.y)

logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

logreg.fit(train_x, train_y)

print('*'*50)

print('regresión logística')

print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {logreg.score(test_x, test_y)}')

print(f'accuracy de Validación: {logreg.score(test_out_x, test_out_y)}')


svc = SVC(gamma='auto')

svc.fit(train_x, train_y)


print('*'*50)

print('maquina de soporte vectorial')

print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {svc.score(test_x, test_y)}')

print(f'accuracy de Validación: {svc.score(test_out_x, test_out_y)}')


random_arbol = RandomForestClassifier()


random_arbol.fit(train_x, train_y)


print('*'*50)

print('arbol random')

print(f'accuracy de Entrenamiento de Entrenamiento: {random_arbol.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {random_arbol.score(test_x, test_y)}')

print(f'accuracy de Validación: {random_arbol.score(test_out_x, test_out_y)}')


arbol = DecisionTreeClassifier()


arbol.fit(train_x, train_y)


print('*'*50)

print('arbol desicion')

print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {arbol.score(test_x, test_y)}')

print(f'accuracy de Validación: {arbol.score(test_out_x, test_out_y)}')