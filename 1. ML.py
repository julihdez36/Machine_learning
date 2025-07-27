# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 14:05:14 2025

@author: Julian
"""

# Machine learning

''''
Artificial intelligence:
    Any technique which enables computers to mimic human behavior (1950)
Machine learning:
    AI techniques that give computers the ability to learn without being explicitly programmed
    to do so (1980).
Deeo learning: 
    AI subset of ML which make the computation of multi-layer neural networks feasible (2010).
    
AI involve different areas or branches, like robotic, speech recognition, expert systems, 
natural language proccessing and machine learning.


Tradicional programming: inputs and outputs through computer an finally you have a program

'''

x = [1,5,2,-2,5]
y = [2,8,4,6,10]

z = []
for i in range(len(x)):
    z.append(x[i]+y[i])
    
import random

# Pero los datos en la vida real tienen un componente estocástico

z_real = []

for i in range(len(x)):
    z_real.append(x[i]+y[i]+random.uniform(0, 1))
    
error = []    
for i in range(len(z)):
    error.append(z[i]-z_real[i])
    

print(f'Errores de predicción {error}')

'''
Taxonomy of machine learning algorithms

Pedro domingos. The master algotihm, 2015.

58:25

Enfoques básicos del ML:
    - Unsupervised learning
    - Supervised learning
    - Reinforcement learning
'''


## Sesión 2
# Tareas ML: clasificación, regressión, clustering
# CRSIP-DM


## KNN classification
# https://martin-thoma.com/k-nearest-neighbor-classification-interactive-example/


import seaborn as sns
iris = sns.load_dataset('iris')
iris.head()

sns.pairplot(iris, hue='species', size=1.5)


iris.columns


sns.scatterplot(data = iris, x = 'sepal_length',
                y = 'petal_length', hue= 'species')

# Definamos matriz de caracteristicas y vector de salida

X = iris.iloc[:,:-1]
y = iris.loc[:,'species']


#################################################
# El criterio de proximidad será la distancia euclidiana

import math
from collections import Counter


# Calcular distancia euclidiana
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# Predecir etiqueta para un punto de prueba
def predict_one(x_train, y_train, x_test, k):
    # Calcular todas las distancias del punto de prueba a los puntos de entrenamiento
    distances = [(euclidean_distance(x_test, x), y) for x, y in zip(x_train, y_train)]
    
    # Ordenar por distancia y elegir los k vecinos más cercanos
    k_neighbors = sorted(distances, key=lambda x: x[0])[:k]
    
    # Extraer las etiquetas de los vecinos y votar la clase mayoritaria
    k_labels = [label for _, label in k_neighbors]
    most_common = Counter(k_labels).most_common(1)[0][0]
    return most_common

# Predecir etiquetas para un conjunto de prueba
def predict(x_train, y_train, x_test, k):
    return [predict_one(x_train, y_train, x, k) for x in x_test]