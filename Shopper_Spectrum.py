# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create EDA_Charts folder
eda_folder = os.path.join(os.path.dirname(__file__), 'EDA_Charts')
os.makedirs(eda_folder, exist_ok=True)

# Step 1 — Dataset Understanding

df = pd.read_csv('online_retail.csv')
print(df.shape)          # rows & columns
print(df.dtypes)         # data types
print(df.isnull().sum()) # missing values
print(df.duplicated().sum()) # duplicates
print(df.describe())

# Step 2 — Data Preprocessing

# 1. Parse dates
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# 2. Drop missing CustomerID
df = df.dropna(subset=['CustomerID'])
df['CustomerID'] = df['CustomerID'].astype(int)

# 3. Remove cancelled invoices
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# 4. Remove invalid quantity/price
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# 5. Drop Duplicates
df = df.drop_duplicates()
print(f"After dropping duplicates: {df.shape}")

# 6. Add TotalPrice column (needed for monetary)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
print(f"Clean dataset: {df.shape}")

# Step 3 — EDA_CHARTS_VISUALIZATION

# 1. Top 10 countries by transaction volume
df['Country'].value_counts().head(10).plot(kind='bar', title='Top 10 Countries')
plt.tight_layout()
plt.savefig(os.path.join(eda_folder, '1_top_10_countries.png'))
plt.show()

# 2. Top 20 selling products
df.groupby('Description')['Quantity'].sum().nlargest(20).plot(kind='barh', title='Top 20 Products')
plt.tight_layout()
plt.savefig(os.path.join(eda_folder, '2_top_20_products.png'))
plt.show()

# 3. Monthly purchase trend
df.set_index('InvoiceDate').resample('ME')['TotalPrice'].sum().plot(title='Monthly Revenue Trend')
plt.tight_layout()
plt.savefig(os.path.join(eda_folder, '3_monthly_revenue_trend.png'))
plt.show()

# 4. Monetary distribution per customer
customer_spend = df.groupby('CustomerID')['TotalPrice'].sum()
customer_spend.hist(bins=50)
plt.title('Customer Spend Distribution')
plt.tight_layout()
plt.savefig(os.path.join(eda_folder, '4_customer_spend_distribution.png'))
plt.show()

# Step 4 — RFM Feature Engineering
# Reference date = day after the last transaction
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    Recency   = ('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
    Frequency = ('InvoiceNo',   'nunique'),
    Monetary  = ('TotalPrice',  'sum')
).reset_index()

# Plot RFM distributions

rfm[['Recency','Frequency','Monetary']].hist(bins=50, figsize=(12,4))
plt.suptitle('RFM Distributions')        
plt.tight_layout()                       
plt.savefig(os.path.join(eda_folder, '4b_rfm_distributions.png'))  
plt.show()                               

# Normalize before clustering:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

# Step 5 — Clustering (KMeans)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Elbow method
inertias = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)

# Elbow curve
plt.plot(range(2, 11), inertias, marker='o')
plt.title('Elbow Curve')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig(os.path.join(eda_folder, '5b_elbow_curve.png'))
plt.show()

# Silhouette scores
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(rfm_scaled)
    print(k, silhouette_score(rfm_scaled, labels))

# Final model
km = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = km.fit_predict(rfm_scaled)

# Interpret cluster means to assign labels
print(rfm.groupby('Cluster')[['Recency','Frequency','Monetary']].mean())
# Map clusters to labels based on the means — e.g., low Recency + high F + high M → High-Value.
# Assign cluster labels
cluster_label_map = {0: 'Regular', 1: 'At-Risk', 2: 'High-Value', 3: 'Occasional'}
rfm['Segment'] = rfm['Cluster'].map(cluster_label_map)
# Save RFM data for Streamlit
rfm.to_csv('rfm_data.csv', index=False)

# 3D scatter plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rfm['Recency'], rfm['Frequency'], rfm['Monetary'],
           c=rfm['Cluster'], cmap='tab10')
ax.set_xlabel('Recency')
ax.set_ylabel('Frequency')
ax.set_zlabel('Monetary')
ax.set_title('Customer Segments - 3D RFM Plot')
plt.tight_layout()
plt.savefig(os.path.join(eda_folder, '5_3d_rfm_clusters.png'))
plt.show()
plt.close('all')   

# Save the model:
import pickle
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(km, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Step 6 — Recommendation System
from sklearn.metrics.pairwise import cosine_similarity

# Build customer-product matrix
pivot = df.groupby(['CustomerID', 'Description'])['Quantity'].sum().unstack().fillna(0)

# Item-item cosine similarity
item_sim = cosine_similarity(pivot.T)
item_sim_df = pd.DataFrame(item_sim, index=pivot.columns, columns=pivot.columns)

# Heatmap of top products
top_products = df['Description'].value_counts().head(20).index
fig, ax = plt.subplots(figsize=(14, 10))   # fresh figure, no 3D conflict
sns.heatmap(item_sim_df.loc[top_products, top_products], ax=ax)
ax.set_title('Product Similarity Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(eda_folder, '6_product_similarity_heatmap.png'))
plt.show()

# Recommendation function
def recommend(product_name, n=5):
    if product_name not in item_sim_df.columns:
        return []
    sim_scores = item_sim_df[product_name].sort_values(ascending=False)
    return sim_scores.iloc[1:n+1].index.tolist()

# Test it
print(recommend('GREEN VINTAGE SPOT BEAKER'))
# Save the similarity matrix:
item_sim_df.to_pickle('item_similarity.pkl')


