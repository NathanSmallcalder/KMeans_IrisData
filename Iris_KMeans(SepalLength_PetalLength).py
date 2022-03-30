from cProfile import label
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans

#Reading the datafile
data = pd.read_csv('iris.csv')

#Setting the x and y axis to scale (0-10)
plt.xlim(0,10)
plt.ylim(0,10)

#Seperating the values of the array
x = data.iloc[:,[0,1,2,3]].values

#Defining the number of clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Plotting Sepal Length Against Petal Length
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 2], s = 50, c = 'purple', label = 'Iris-versicolor')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 2], s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 2], s = 100, c = 'green', label = 'Iris-virginica')
plt.ylabel('Petal Length')
plt.xlabel('Sepal Length')

#Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,2], s = 100, c = 'red', label = 'Centroids')


plt.legend()
plt.show()

