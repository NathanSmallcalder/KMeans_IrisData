import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
import csv

with open('iris.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

SepalLength_Setosa = []
SepalLength_Versicolor = []
SepalLength_Virginica =[]

PetalLength_Setosa = []
PetalLength_Versicolor = []
PetalLength_Virginica =[]

i = 0
#Organising SepalLength Data
while i < len(data) -1:
    if(data[i][4] == 'Iris-setosa'):
        SepalLength_Setosa.append(data[i][0])
    if(data[i][4] == 'Iris-versicolor'):
        SepalLength_Versicolor.append(data[i][0])
    if(data[i][4] == 'Iris-virginica'):
        SepalLength_Virginica.append(data[i][0])
    i = i + 1

i = 0
#Organising PetalLength Data
while i < len(data) -1:
    if(data[i][4] == 'Iris-setosa'):
        PetalLength_Setosa.append(data[i][2])
    if(data[i][4] == 'Iris-versicolor'):
        PetalLength_Versicolor.append(data[i][2])
    if(data[i][4] == 'Iris-virginica'):
        PetalLength_Virginica.append(data[i][2])
    i = i + 1

print(SepalLength_Setosa)
X, y = make_blobs(n_samples=5, centers=4, cluster_std=0.60, random_state=0)
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
plt.scatter(PetalLength_Setosa,SepalLength_Setosa)
plt.scatter(PetalLength_Versicolor,PetalLength_Versicolor)
plt.scatter(SepalLength_Virginica,PetalLength_Virginica)
plt.show()