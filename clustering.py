import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv(
    "/Users/aman/Desktop/Work/2020-21/dwm/datasets/mall_customers.csv")
print(data.head())
print(data.isnull().any())
 
data = data.iloc[:, [2, 3, 4]].values
print(data.shape)

distortions = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=0)
    model = kmeans.fit(data)
    distortions.append(kmeans.inertia_)
 
plt.plot(range(2, 10), distortions)
plt.xlabel("number of clusters")
plt.ylabel("metric")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=0)
model = kmeans.fit(data)
predictions = model.predict(data)
 
centers = model.cluster_centers_

plt.scatter(data[:, 0], data[:, 1], c=predictions, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel("age")
plt.ylabel("annual income")
plt.show()

plt.scatter(data[:, 0], data[:, 1], c=predictions, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel("age")
plt.ylabel("annual income")
plt.show()

plt.scatter(data[:, 0], data[:, 2], c=predictions, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 2], c='black', s=200, alpha=0.5)
plt.xlabel("age")
plt.ylabel("spending score")
plt.show()
 
from mpl_toolkits.mplot3d import Axes3D
 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=predictions)
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=75, alpha=0.5)
plt.show()