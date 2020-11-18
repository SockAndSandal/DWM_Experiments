import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

data = pd.read_csv("/Users/aman/Desktop/Work/2020-21/dwm/datasets/mall_customers.csv")
print(data.head())
print(data.isnull().any())
 
data = data.iloc[:, [2, 3, 4]].values
print(data.shape)

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
 
plt.title('Hierarchical Clustering Dendrogram')
plot_dendrogram(model, truncate_mode='level', p=5)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")

agglo = AgglomerativeClustering(n_clusters=5)
predictions = agglo.fit_predict(data)

plt.scatter(data[:, 0], data[:, 1], c=predictions, s=50, cmap='viridis')
plt.xlabel("age")
plt.ylabel("annual income")
plt.show()

plt.scatter(data[:, 0], data[:, 1], c=predictions, s=50, cmap='viridis')
plt.xlabel("age")
plt.ylabel("annual income")
plt.show()
 
plt.scatter(data[:, 0], data[:, 2], c=predictions, s=50, cmap='viridis')
plt.xlabel("age")
plt.ylabel("spending score")
plt.show()
 
from mpl_toolkits.mplot3d import Axes3D
 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=predictions)
plt.show()
