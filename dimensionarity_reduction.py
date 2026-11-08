from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

# UMAP 
def dim_reduction_UMAP(df_reviews):
    # 正規化
    scaler =  MinMaxScaler()
    df_reviews_scaled = scaler.fit_transform(df_reviews)

    umap = UMAP(n_components=290, metric ='cosine', min_dist = 1, angular_rp_forest = True, n_neighbors = 110)

    df_reviews_transformed = umap.fit_transform(df_reviews_scaled)

    return df_reviews_transformed