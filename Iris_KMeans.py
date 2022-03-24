from cProfile import label
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import preprocessing

data = pd.read_csv('iris.csv')

plt.xlim(0,10)
plt.ylim(0,10)

x = data.iloc[:,lambda df: [0,3]]
print(x)
kmeans = KMeans(2)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
print(identified_clusters)

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['SepalLength'],data_with_clusters['PetalWidth'],c=data_with_clusters['Clusters'],cmap='rainbow')
print(kmeans.cluster_centers_[:,1])
plt.scatter(kmeans.cluster_centers_[:,0]
, kmeans.cluster_centers_[:,1],s = 100,c = 'yellow'
, label = 'Centroids')

plt.xlabel('SepalLength')
plt.ylabel('PetalWidth')
plt.show()