import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

simplefilter(action='ignore', category=FutureWarning)

datos = pd.read_csv('diabetes.csv')

datos.Age.replace(np.nan, 33, inplace=True)

rangos = [20, 35, 50, 60, 80, 100]
nombres = ['1', '2', '3', '4', '5']

datos.Age = pd.cut(datos.Age, rangos, labels=nombres)
datos.drop(['DiabetesPedigreeFunction', 'BMI', 'Insulin', 'BloodPressure'], axis=1, inplace=True)


train_data = datos[:384]
test_data = datos[384:]


x = np.array(train_data.drop(['Outcome'], 1))
y = np.array(train_data.Outcome) 


train_x, test_x, train_y, test_y = train_test_split(x, y, size_test=0.2)

test_out_x = np.array(test_data.drop(['Outcome'], 1))
test_out_y = np.array(test_data.Outcome) 

logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)


logreg.fit(train_x, train_y)


print('*'*50)

print('regresión Logística')

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


arbol = DecisionTreeClassifier()


arbol.fit(train_x, train_y)


print('*'*50)

print('arbol de decision')

print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {arbol.score(test_x, test_y)}')

print(f'accuracy de Validación: {arbol.score(test_out_x, test_out_y)}')


arbol_random = RandomForestClassifier()

arbol_random.fit(train_x, train_y)


print('*'*50)

print('arbol random')

print(f'accuracy de Entrenamiento de Entrenamiento: {arbol_random.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {arbol_random.score(test_x, test_y)}')

print(f'accuracy de Validación: {arbol_random.score(test_out_x, test_out_y)}')

Knn = KNeighborsClassifier()

Knn.fit(train_x, train_y)


print('*'*50)

print('K-nearest neighbors')

print(f'accuracy de Entrenamiento de Entrenamiento: {Knn.score(train_x, train_y)}')

print(f'accuracy de Test de Entrenamiento: {Knn.score(test_x, test_y)}')

print(f'accuracy de Validación: {Knn.score(test_out_x, test_out_y)}')