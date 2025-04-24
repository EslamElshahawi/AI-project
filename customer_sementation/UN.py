import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt   
import seaborn as sns
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
mall = pd.read_csv('D:\Mall_Customers.csv')
mall.head()

mall.info()
X = mall.iloc[:, [3,4]]
X

plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'])

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42, init='k-means++')

kmeans.fit(X)
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_
y=mall['Cluster'] = kmeans.fit_predict(X)
y
mall

plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=kmeans.labels_, cmap ='rainbow') 
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
wcss    
    
plt.plot(range(1,11),wcss)
plt.title("ELBOW METHOD")
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=kmeans.labels_, cmap ='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='k', s=200)