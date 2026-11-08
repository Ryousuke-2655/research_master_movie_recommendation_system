# モデル比較参考　「Exploring Dimensionality Reduction Techniques in Multilingual Transformers」
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import KernelPCA
from umap import UMAP

from test_user_candidates import user_candidates_selection, verification, build_user_indices, prepare_user_splits_and_neighbors
from wilcoxon import maeAndRmse_wilcoxon

#デフォルト：n_ncomponent=100
#IPCA（漸進的主成分分析）
def dim_reduction_IPCA(df_reviews):
    # 標準化
    scaler = StandardScaler()
    df_reviews_scaled = scaler.fit_transform(df_reviews)

    ipca = IncrementalPCA(n_components=100)

    df_reviews_transformed = ipca.fit_transform(df_reviews_scaled)
    return df_reviews_transformed

#ICA（独立成分分析）
def dim_reduction_ICA(df_reviews):
    ica = FastICA(n_components=100, random_state = 0, max_iter= 320, whiten = 'arbitrary-variance', tol = 5e-4 )

    df_reviews_transformed = ica.fit_transform(df_reviews)

    return df_reviews_transformed 

# KPCA
#  sigmoid
def dim_reduction_KPCA_sigmoid(df_reviews):
    # 標準化
    scaler = StandardScaler()
    df_reviews_scaled = scaler.fit_transform(df_reviews)

    kpca_sig = KernelPCA(n_components=100, kernel='sigmoid', eigen_solver = 'arpack',copy_X = False, random_state = 42)

    df_reviews_transformed = kpca_sig.fit_transform(df_reviews_scaled)
    return df_reviews_transformed 

# polynomial
def dim_reduction_KPCA_polynomial(df_reviews):
    # 標準化
    scaler = StandardScaler()
    df_reviews_scaled = scaler.fit_transform(df_reviews)
    
    kpca_poly = KernelPCA(n_components=100, kernel='poly', eigen_solver = 'arpack',copy_X = False, random_state = 42)

    df_reviews_transformed = kpca_poly.fit_transform(df_reviews_scaled)

    return df_reviews_transformed

# rbf
def dim_reduction_KPCA_rbf(df_reviews):
    # 標準化
    scaler = StandardScaler()
    df_reviews_scaled = scaler.fit_transform(df_reviews)
    
    kpca_rbf = KernelPCA(n_components=100, kernel='rbf', eigen_solver = 'arpack',copy_X = False, random_state = 42)

    df_reviews_transformed = kpca_rbf.fit_transform(df_reviews_scaled)

    return df_reviews_transformed

# cosine
def dim_reduction_KPCA_cosine(df_reviews):
    # 標準化
    scaler = StandardScaler()
    df_reviews_scaled = scaler.fit_transform(df_reviews)
    
    kpca_sig = KernelPCA(n_components=100, kernel='cosine', eigen_solver = 'arpack',copy_X = False, random_state = 42)

    df_reviews_transformed = kpca_sig.fit_transform(df_reviews_scaled)
    return df_reviews_transformed 

# VarianceThreshold（変動閾値）
def dim_reduction_VarianceThreshold(df_reviews, num):
    df_reviews = df_reviews.astype(np.float32)
    # 正規化
    scaler =  MinMaxScaler()
    df_reviews_scaled = scaler.fit_transform(df_reviews)

    variances = np.var(df_reviews_scaled, axis=0)

    percentile = 10 * min(max(num, 1), 9)
    deciles_threshold = np.percentile(variances, percentile)

    mask = variances > deciles_threshold
    if not mask.any():
        print(f"Threshold {deciles_threshold:.6f} -> All features removed")
        return None
    return df_reviews_scaled[:, mask]

# UMAP 
def dim_reduction_UMAP(df_reviews, n_components = 100, n_neighbors = 110):
    # 正規化
    scaler =  MinMaxScaler()
    df_reviews_scaled = scaler.fit_transform(df_reviews)

    umap = UMAP(n_components = n_components, metric ='cosine', min_dist = 1, angular_rp_forest = True, n_neighbors = n_neighbors)

    df_reviews_transformed = umap.fit_transform(df_reviews_scaled)

    return df_reviews_transformed

def dimensionarity_reduction_select(name, df_reviews):
    match name:
        case 'normal':
            return df_reviews
        case 'IPCA':
            return dim_reduction_IPCA(df_reviews)
        case 'ICA':
            return dim_reduction_ICA(df_reviews)
        case 'KPCA_sigmoid':
            return dim_reduction_KPCA_sigmoid(df_reviews)
        case 'KPCA_polynomial':
            return dim_reduction_KPCA_polynomial(df_reviews)
        case 'KPCA_rbf':
            return dim_reduction_KPCA_rbf(df_reviews)
        case 'KPCA_cosine':
            return dim_reduction_KPCA_cosine(df_reviews)
        case 'VarianceThreshold':
            return dim_reduction_VarianceThreshold(df_reviews, 1) 
        case 'UMAP':
            return dim_reduction_UMAP(df_reviews)


# 次元削減手法の比較検証
# def main():
#     # ファイルパス
#     ratings_file_path = 'data/updated_ratings.json'
#     reviews_file_path = 'data/review_mean_embeddings_per_movie.csv'
    
#     per_MeanMae_list = []
#     per_MeanRmse_list = []
#     model_mae_list =[]
#     model_rmse_list =[]
#     method_name = ['normal', 'IPCA', 'ICA', 'KPCA_sigmoid', 'KPCA_polynomial', 'KPCA_rbf', 'KPCA_cosine', 'VarianceThreshold', 'UMAP']
#     division_rate =0.3
#     num_test_users = 100

#     # ファイル読み込み
#     df_reviews_per_movie = pd.read_csv(reviews_file_path)
#     if os.path.exists(ratings_file_path):
#         df_ratings = pd.read_json(ratings_file_path, lines=True)
#     else:
#         print(f"ファイルが見つかりません: {ratings_file_path}")

#     df_re = df_reviews_per_movie.drop(columns=['item_id'])
#     notre = df_reviews_per_movie['item_id']
#     df_ratings_ex = df_ratings.drop(columns=['tmdbId', 'title'])

#     #テストユーザー抽出
#     test_users = user_candidates_selection(df_ratings_ex, num_test_users)
#     exclude_users = set(test_users)
    
#     other_user_data = df_ratings_ex[
#         ~df_ratings_ex['user_id'].isin(exclude_users)
#     ]
#     user_ratings_map, user_rows_map, item_to_users = build_user_indices(other_user_data)
#     ratings_by_user = {uid: g.drop(columns=['user_id']) for uid, g in df_ratings_ex.groupby('user_id')}

#     prepared = prepare_user_splits_and_neighbors(test_users, ratings_by_user, division_rate, user_ratings_map, user_rows_map, item_to_users)

#     for i,n in enumerate(method_name):
#         print(n)
#         df_re_dimre =dimensionarity_reduction_select(n, df_re)

#         df_notre=pd.DataFrame(notre)
#         df_re_dimre_new =pd.DataFrame(df_re_dimre)
#         df_reviews_per_movies_dim = pd.concat([df_notre, df_re_dimre_new], axis=1)
#         mae,rmse, df_metrics = verification(df_reviews_per_movies_dim, prepared)
#         model_mae_list.append(mae)
#         model_rmse_list.append(rmse)
#         mean_mae = np.mean(mae)
#         mean_rmse = np.mean(rmse)
#         per_MeanMae_list.append(mean_mae)
#         per_MeanRmse_list.append(mean_rmse)
    
#     column_names = [f"col_{i}" for i in range(num_test_users)]
#     df_model_mae = pd.DataFrame(model_mae_list, columns=column_names)
#     df_model_rmse = pd.DataFrame(model_rmse_list, columns=column_names)
#     df_model_mae.to_csv(f"result/dimensionarity_reduction_model_name_nae.csv", index=False, encoding="utf-8-sig")
#     df_model_rmse.to_csv(f"result/dimensionarity_reduction_model_name_rmse.csv", index=False, encoding="utf-8-sig")
#     df_csv = pd.DataFrame({"method_name":method_name,"MAE": per_MeanMae_list, "RMSE": per_MeanRmse_list})
#     df_csv.to_csv(f"result/dimensionarity_reduction_name.csv", index=False, encoding="utf-8-sig")

#     stat_mae, stat_rmse, p_mae, p_rmse, max_mean_mae_index, second_max_mean_mae_index, max_mean_rmse_index, second_max_mean_rmse_index = maeAndRmse_wilcoxon(model_mae_list, model_rmse_list, per_MeanMae_list, per_MeanRmse_list)
#     df_csv_wilcoxon = pd.DataFrame({"stat_mae":stat_mae,"stat_rmse": stat_rmse, "p_mae": p_mae,"p_rmse": p_rmse, "max_mean_mae_index": max_mean_mae_index, "second_max_mean_mae_index":second_max_mean_mae_index,"max_mean_rmse_index": max_mean_rmse_index, "second_max_mean_rmse_index": second_max_mean_rmse_index}, index=[0])
#     df_csv_wilcoxon.to_csv(f"result/dimensionarity_reduction_wilcoxon.csv", index=False, encoding="utf-8-sig", float_format="%.10e")

# 変動閾値の比較検証
# def main():
#     # ファイルパス
#     ratings_file_path = 'data/updated_ratings.json'
#     reviews_file_path = 'data/review_mean_embeddings_per_movie.csv'

#     per_MeanMae_list = []
#     per_MeanRmse_list = []
#     all_thresholds = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
#     division_rate =0.3
#     num_test_users = 100

#     # ファイル読み込み
#     df_reviews_per_movie = pd.read_csv(reviews_file_path)
#     if os.path.exists(ratings_file_path):
#         df_ratings = pd.read_json(ratings_file_path, lines=True)
#     else:
#         print(f"ファイルが見つかりません: {ratings_file_path}")
    
#     df_re = df_reviews_per_movie.drop(columns=['item_id'])
#     notre = df_reviews_per_movie['item_id']
#     df_ratings_ex = df_ratings.drop(columns=['tmdbId', 'title'])

#     #テストユーザー抽出
#     test_users = user_candidates_selection(df_ratings_ex, num_test_users)
#     exclude_users = set(test_users)
    
#     other_user_data = df_ratings_ex[
#         ~df_ratings_ex['user_id'].isin(exclude_users)
#     ]
#     user_ratings_map, user_rows_map, item_to_users = build_user_indices(other_user_data)
#     ratings_by_user = {uid: g.drop(columns=['user_id']) for uid, g in df_ratings_ex.groupby('user_id')}

#     prepared = prepare_user_splits_and_neighbors(test_users, ratings_by_user, division_rate, user_ratings_map, user_rows_map, item_to_users)

#     for i,n in enumerate(all_thresholds):
#         print(n)
#         df_re_dimre = dim_reduction_VarianceThreshold(df_re,i+1)
#         df_notre=pd.DataFrame(notre)
#         df_re_dimre_new =pd.DataFrame(df_re_dimre)
#         df_reviews_per_movies_dim = pd.concat([df_notre, df_re_dimre_new], axis=1)
#         mae,rmse, df_metrics = verification(df_reviews_per_movies_dim, prepared)
#         mean_mae = np.mean(mae)
#         mean_rmse = np.mean(rmse)
#         per_MeanMae_list.append(mean_mae)
#         per_MeanRmse_list.append(mean_rmse)
#         if i ==0:
#             print('クリア')
    
#     df_csv = pd.DataFrame({"all_thresholds":all_thresholds,"MAE": per_MeanMae_list, "RMSE": per_MeanRmse_list})
    
#     df_csv.to_csv(f"result/varianceThreshold.csv", index=False, encoding="utf-8-sig")
    
#     plt.plot(all_thresholds, per_MeanMae_list, marker="o")
#     plt.title('thresholdsごとのMAE')
#     plt.xlabel('thresholds')
#     plt.ylabel('MAE')
#     plt.xlim(0, None) 
#     plt.ylim(0, None) 
#     plt.savefig('result/mae_varianceThreshold.png')
#     plt.show()

#     plt.plot(all_thresholds, per_MeanRmse_list, marker="o")
#     plt.title('thresholdsごとのRMSE')
#     plt.xlabel('thresholds')
#     plt.ylabel('RMSE')
#     plt.xlim(0, None) 
#     plt.ylim(0, None) 
#     plt.savefig('result/rmse_varianceThreshold.png')
#     plt.show()

# UMAPの近傍の数の比較検証
# def main():
#     # ファイルパス
#     ratings_file_path = 'data/updated_ratings.json'
#     reviews_file_path = 'data/review_mean_embeddings_per_movie.csv'

#     per_MeanMae_list = []
#     per_MeanRmse_list = []
#     umap_n_neighbors  = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 125]
#     division_rate =0.3
#     num_test_users = 100

#     # ファイル読み込み
#     df_reviews_per_movie = pd.read_csv(reviews_file_path)
#     if os.path.exists(ratings_file_path):
#         df_ratings = pd.read_json(ratings_file_path, lines=True)
#     else:
#         print(f"ファイルが見つかりません: {ratings_file_path}")
    
#     df_re = df_reviews_per_movie.drop(columns=['item_id'])
#     notre = df_reviews_per_movie['item_id']
#     df_ratings_ex = df_ratings.drop(columns=['tmdbId', 'title'])

#     #テストユーザー抽出
#     test_users = user_candidates_selection(df_ratings_ex, num_test_users)
#     exclude_users = set(test_users)
    
#     other_user_data = df_ratings_ex[
#         ~df_ratings_ex['user_id'].isin(exclude_users)
#     ]
#     user_ratings_map, user_rows_map, item_to_users = build_user_indices(other_user_data)
#     ratings_by_user = {uid: g.drop(columns=['user_id']) for uid, g in df_ratings_ex.groupby('user_id')}

#     prepared = prepare_user_splits_and_neighbors(test_users, ratings_by_user, division_rate, user_ratings_map, user_rows_map, item_to_users)

#     for i,n in enumerate(umap_n_neighbors):
#         print(n)
#         df_re_dimre = dim_reduction_UMAP(df_re, 100 , n)
#         df_notre=pd.DataFrame(notre)
#         df_re_dimre_new =pd.DataFrame(df_re_dimre)
#         df_reviews_per_movies_dim = pd.concat([df_notre, df_re_dimre_new], axis=1)
#         mae,rmse, df_metrics = verification(df_reviews_per_movies_dim, prepared)
#         mean_mae = np.mean(mae)
#         mean_rmse = np.mean(rmse)
#         per_MeanMae_list.append(mean_mae)
#         per_MeanRmse_list.append(mean_rmse)
#         if i ==0:
#             print('クリア')
    
#     df_csv = pd.DataFrame({"umap_n_neighbors":umap_n_neighbors,"MAE": per_MeanMae_list, "RMSE": per_MeanRmse_list})
    
#     df_csv.to_csv(f"result/umap_n_neighbors.csv", index=False, encoding="utf-8-sig")
    
#     plt.plot(umap_n_neighbors, per_MeanMae_list, marker="o")
#     plt.title('n_neighborsごとのMAE')
#     plt.xlabel('n_neighbors')
#     plt.ylabel('MAE')
#     plt.xlim(0, None) 
#     plt.ylim(0, None) 
#     plt.savefig('result/mae_umap_n_neighbors.png')
#     plt.show()

#     plt.plot(umap_n_neighbors, per_MeanRmse_list, marker="o")
#     plt.title('n_neighborsごとのRMSE')
#     plt.xlabel('n_neighbors')
#     plt.ylabel('RMSE')
#     plt.xlim(0, None) 
#     plt.ylim(0, None) 
#     plt.savefig('result/rmse_umap_n_neighbors.png')
#     plt.show()

# n_componentsの比較検証
# def main():
#     # ファイルパス
#     ratings_file_path = 'data/updated_ratings.json'
#     reviews_file_path = 'data/review_mean_embeddings_per_movie.csv'

#     per_MeanMae_list = []
#     per_MeanRmse_list = []
#     n_components  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380]
#     division_rate =0.3
#     num_test_users = 100

#     # ファイル読み込み
#     df_reviews_per_movie = pd.read_csv(reviews_file_path)
#     if os.path.exists(ratings_file_path):
#         df_ratings = pd.read_json(ratings_file_path, lines=True)
#     else:
#         print(f"ファイルが見つかりません: {ratings_file_path}")
    
#     df_re = df_reviews_per_movie.drop(columns=['item_id'])
#     notre = df_reviews_per_movie['item_id']
#     df_ratings_ex = df_ratings.drop(columns=['tmdbId', 'title'])

#     #テストユーザー抽出
#     test_users = user_candidates_selection(df_ratings_ex, num_test_users)
#     exclude_users = set(test_users)
    
#     other_user_data = df_ratings_ex[
#         ~df_ratings_ex['user_id'].isin(exclude_users)
#     ]
#     user_ratings_map, user_rows_map, item_to_users = build_user_indices(other_user_data)
#     ratings_by_user = {uid: g.drop(columns=['user_id']) for uid, g in df_ratings_ex.groupby('user_id')}

#     prepared = prepare_user_splits_and_neighbors(test_users, ratings_by_user, division_rate, user_ratings_map, user_rows_map, item_to_users)

#     for i,n in enumerate(n_components):
#         print(n)
#         df_re_dimre = dim_reduction_UMAP(df_re, n)
#         df_notre=pd.DataFrame(notre)
#         df_re_dimre_new =pd.DataFrame(df_re_dimre)
#         df_reviews_per_movies_dim = pd.concat([df_notre, df_re_dimre_new], axis=1)
#         mae,rmse, df_metrics = verification(df_reviews_per_movies_dim, prepared)
#         mean_mae = np.mean(mae)
#         mean_rmse = np.mean(rmse)
#         per_MeanMae_list.append(mean_mae)
#         per_MeanRmse_list.append(mean_rmse)
#         if i ==0:
#             print('クリア')
    
#     df_csv = pd.DataFrame({"n_components":n_components,"MAE": per_MeanMae_list, "RMSE": per_MeanRmse_list})
    
#     df_csv.to_csv(f"result/n_components.csv", index=False, encoding="utf-8-sig")
    
#     plt.plot(n_components, per_MeanMae_list, marker="o")
#     plt.title('n_componentsごとのMAE')
#     plt.xlabel('n_components')
#     plt.ylabel('MAE')
#     plt.xlim(0, None) 
#     plt.ylim(0, 2) 
#     plt.savefig('result/mae_n_components.png')
#     plt.show()

#     plt.plot(n_components, per_MeanRmse_list, marker="o")
#     plt.title('n_componentsごとのRMSE')
#     plt.xlabel('n_components')
#     plt.ylabel('RMSE')
#     plt.xlim(0, None) 
#     plt.ylim(0, 2) 
#     plt.savefig('result/rmse_n_components.png')
#     plt.show()

# if __name__ == "__main__":
#     main()