from cProfile import label
from optparse import Values
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import preprocessing


data = pd.read_csv('iris.csv')


x = data.iloc[:,[0,1,2,3]].values
print(x)

wcss=[]
for i in range(1,11):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)
   
   
print(wcss)
number_clusters = range(1,11)
plt.plot(number_clusters,wcss)
plt.title('Elbow Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()