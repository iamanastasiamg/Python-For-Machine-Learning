import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def drop_column(df, column_name, inplace=True):
    return df.drop([column_name], axis='columns', inplace=inplace)

# Fit KMeans clustering
def apply_KMeans(K, data):
    kmeans = KMeans(
        n_clusters=K,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42
    )
    kmeans.fit(data)
    # Obtain cluster memberships for each item in the data
    y_kmeans = kmeans.predict(data)
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    for k in range(K):
        plt.scatter(data[y_kmeans == k, 0], data[y_kmeans == k, 1], s=100, label="k = " + str(k + 1))
    #Add cluster centroids to the plot
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=100, label='Centroids')
    plt.title("KMeans Clustering Scatter Plot")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.legend()
    plt.show()

# 1.Load the Dataset
#Load the Mall Customers dataset and print first 5 rows.
df = pd.read_csv("dataset/mall_customers.csv")
print(df.head())

# 2.Perform Basic Preprocessing
# Calculate IQR for Annual Income and Spending Score
Q1 = df[['Annual Income (k$)', 'Spending Score (1-100)']].quantile(0.25)
Q3 = df[['Annual Income (k$)', 'Spending Score (1-100)']].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for non-outlier values
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers based on IQR
df_no_outliers_iqr = df[
    (df['Annual Income (k$)'] >= lower_bound['Annual Income (k$)']) &
    (df['Annual Income (k$)'] <= upper_bound['Annual Income (k$)']) &
    (df['Spending Score (1-100)'] >= lower_bound['Spending Score (1-100)']) &
    (df['Spending Score (1-100)'] <= upper_bound['Spending Score (1-100)'])
]
#Visualize the results
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
# Plot Original Data
ax[0].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c='blue', label='Original Data', edgecolors='k', s=100)
ax[0].set_title('Original Data')
ax[0].set_xlabel('Annual Income (k$)')
ax[0].set_ylabel('Spending Score (1-100)')
ax[0].legend()
# Plot Data after IQR Removal
ax[1].scatter(df_no_outliers_iqr['Annual Income (k$)'], df_no_outliers_iqr['Spending Score (1-100)'], c='red', label='IQR Cleaned', edgecolors='k', s=100)
ax[1].set_title('After IQR Outlier Removal')
ax[1].set_xlabel('Annual Income (k$)')
ax[1].set_ylabel('Spending Score (1-100)')
ax[1].legend()
plt.tight_layout()
plt.show()

# 3.Analyze Correlation and Feature Selection
features = ['Gender', 'Age', 'Spending Score (1-100)']
# Convert Gender to numeric (0 for Male, 1 for Female)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
corr_matrix = df[features].corr()
#Define the colormap which maps the data values to the color space defined with the diverging_palette method
colors = sns.diverging_palette(150, 275, s=80, l=55, n=9, as_cmap=True)
#Creating heatmap on correlation matrix, set colormap to cmap
sns.heatmap(corr_matrix, annot=True, center=0, cmap=colors, robust=True, fmt='.3g')
plt.show()

# 4.Use KMeans for Clustering
#From same data keep the Annual Income and Spending Score columns
data = np.column_stack((df['Annual Income (k$)'], df['Spending Score (1-100)']))
# Apply KMeans clustering
#for i in range(1, 6):
#    apply_KMeans(i, data)

#5. Evaluate kappa Using Both Methods
#Creating values for the elbow
X = df.loc[:,["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
inertia = []
k = range(1,20)
for i in k:
    means_k = KMeans(n_clusters=i, random_state=0)
    means_k.fit(X)
    inertia.append(means_k.inertia_)
#Plotting the elbow
plt.plot(k , inertia , 'bo-')
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()