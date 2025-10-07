# ======= CLUSTERING ANALYSIS =======
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the results from your previous analysis
per_model = pd.read_csv("/Users/alaamiari/Desktop/ev_results/ev_car_ranking.csv")

# Prepare features for clustering
X = per_model[["score_price", "score_battery", "score_age"]].fillna(0)

# Create clusters (you can change n_clusters to 3, 4, etc.)
kmeans = KMeans(n_clusters=3, random_state=42)
per_model["cluster"] = kmeans.fit_predict(X)

# Show cluster centers
print("\nCluster Centers (mean feature values):")
print(pd.DataFrame(kmeans.cluster_centers_, columns=X.columns))

# Visualize the clusters
sns.pairplot(per_model, hue="cluster", palette="viridis")
plt.suptitle("Car Clusters (by Value Features)", y=1.02)
plt.show()

# Optional: save clustered file
per_model.to_csv("/Users/alaamiari/Desktop/ev_results/ev_car_clusters.csv", index=False)
print("\n✅ Saved file with clusters: ev_car_clusters.csv")
