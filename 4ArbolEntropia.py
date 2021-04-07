# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:42:25 2021

@author: Alexander Humberto Nina Pacajes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv('voice.csv')
x=dataset.drop('label',axis=1)
y=dataset['label']
#x todos features y la col de la clase
#creamos los de entrenamiento y los de prediccion aqui 80% para entrenamiento
list1=[]
for i in range(20):
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.20, random_state=1)
    classifier=DecisionTreeClassifier(criterion='entropy')
    classifier.fit(x_train, y_train)
    
    #para predecir el x_test comparamos luego con y_test
    y_pred=classifier.predict(x_test)
    list1.append(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    #print(classification_report(y_test, y_pred))
    print("----------------")

print(list1)
fin = np.asarray(list1)
media=fin.mean()
mediana=np.median(fin)
print("Media: "+ str(media))
print("Mediana: "+ str(mediana))
print("---------------------------")
print("MediaR: "+ str(round(media,3)))
print("MedianaR: "+ str(round(mediana,3)))


# =============================================================================
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
# X_set, y_set = sc.inverse_transform(x_train), y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('rojo', 'verde')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('rojo', 'verde'))(i), label = j)
# 
# #plt.legend()
# plt.show()
# 
# =============================================================================


