import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv('data/customer_data.csv')

# Drop Customer ID
df.drop('Customer ID', axis=1, inplace=True)

# ----------------------------
# Data Preprocessing
# ----------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# ----------------------------
# Determine Optimal Clusters Using Elbow Method
# ----------------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_plot.png')
plt.show()

# ----------------------------
# Apply K-Means Clustering
# ----------------------------
optimal_clusters = 5  # Change based on Elbow Plot
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
cluster_labels = kmeans.fit_predict(scaled_df)

# Add Cluster Labels to Original Data
df['Cluster'] = cluster_labels
df.to_csv('customer_segmented_data.csv', index=False)
print("Clustered dataset saved as 'customer_segmented_data.csv'")

# ----------------------------
# Visualize Clusters using PCA
# ----------------------------
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
pca_df['Cluster'] = cluster_labels

# Plot Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
plt.title('Customer Segmentation using PCA')
plt.savefig('cluster_plot.png')
plt.show()

# ----------------------------
# Recommendations and Insights
# ----------------------------
print("\n✅ Recommendations:")
print("1. High-spending customers should be targeted for loyalty programs.")
print("2. Middle-income groups may be targeted for promotions.")
print("3. Tailor marketing strategies based on age and income segmentation.")
