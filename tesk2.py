import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture

#Load data
x = np.load('X_train_regression2.npy')
y = np.load('y_train_regression2.npy')

#Initial Regression
reg = LinearRegression().fit(x, y)
pred_y = reg.predict(x)
sse = ((y - pred_y)**2)
#print(sse)

#Remove outliers
#outliers = np.where(sse > 0.01)
#x1 = np.delete(x.copy(), outliers, axis=0)
#y1 = np.delete(y.copy(), outliers, axis=0)
#x2 = x[outliers]
#y2 = y[outliers]


#Cluster doido
n_clusters = 2
#KMeans clustering model
#kmeans = KMeans(n_clusters=n_clusters,init = 'k-means++', n_init = 'auto', random_state=42)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(x)
# Get the cluster assignments for each data point
cluster_assignments = kmeans.labels_
# Split your data into two sets based on the cluster assignments
x_1 = x[cluster_assignments == 0]
x_2 = x[cluster_assignments == 1]
print("------------------")
print(x_1.shape)
print("------------------")
print(x_2.shape)

# Assuming you have 'x' and 'y' as your data
# Stack 'x' and 'y' horizontally to create a single feature matrix
X = np.column_stack((x, y))

# Choose the number of clusters (in this case, 2)
n_clusters = 2

# Create a Gaussian Mixture Model with 'n_clusters'
gmm = GaussianMixture(n_components=n_clusters, random_state=42)

# Fit the GMM to your data
gmm.fit(X)

# Get the cluster assignments for each data point
cluster_assignments = gmm.predict(X)

# Split your data into two sets based on the cluster assignments
data_set_1 = X[cluster_assignments == 0]
data_set_2 = X[cluster_assignments == 1]






