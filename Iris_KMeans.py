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

x = data.iloc[:,1:3]
kmeans = KMeans(3)
kmeans.fit(x)
identified_clusters = kmeans.fit_predict(x)
print(identified_clusters)

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['SepalWidth'],data_with_clusters['PetalLength'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.xlabel('Sepal Width')
plt.ylabel('Petal Length')
plt.show()