import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import japanize_matplotlib
from dimensionarity_reduction import dim_reduction_UMAP

from test_user_candidates import verification_new, user_candidates_selection, build_user_indices, prepare_user_splits_and_neighbors
from wilcoxon import maeAndRmse_wilcoxon

def main():
    # ファイルパス
    ratings_file_path = 'data/updated_ratings.json'
    reviews_file_path = 'data/review_mean_embeddings_per_movie.csv'

    # 初期値
    per_MeanMae_list = []
    per_MeanRmse_list = []
    model_mae_list =[]
    model_rmse_list =[]
    model_name_list = ['ran', 'xg', 'mlp', 'svm', 'knn']
    division_rate =0.3
    num_test_users = 1

    # ファイル読み込み
    df_reviews_per_movie = pd.read_csv(reviews_file_path)
    if os.path.exists(ratings_file_path):
        df_ratings = pd.read_json(ratings_file_path, lines=True)
    else:
        print(f"ファイルが見つかりません: {ratings_file_path}")
    
    df_re = df_reviews_per_movie.drop(columns=['item_id'])
    df_re_dimre =dim_reduction_UMAP(df_re, 100) # 精度がいい次元削減手法選択　n_componentsも入れる
    notre = df_reviews_per_movie['item_id']
    df_ratings_ex = df_ratings.drop(columns=['tmdbId', 'title'])

    #テストユーザー抽出
    test_users = user_candidates_selection(df_ratings_ex, num_test_users)
    exclude_users = set(test_users)
    
    other_user_data = df_ratings_ex[
        ~df_ratings_ex['user_id'].isin(exclude_users)
    ]
    user_ratings_map, user_rows_map, item_to_users = build_user_indices(other_user_data)
    ratings_by_user = {uid: g.drop(columns=['user_id']) for uid, g in df_ratings_ex.groupby('user_id')}

    prepared = prepare_user_splits_and_neighbors(test_users, ratings_by_user, division_rate, user_ratings_map, user_rows_map, item_to_users)

    for i,n in enumerate(model_name_list):
        print(n)
        df_notre=pd.DataFrame(notre)
        df_re_dimre_new =pd.DataFrame(df_re_dimre)
        df_reviews_per_movies_dim = pd.concat([df_notre, df_re_dimre_new], axis=1)
        num_evaluated_users,mae,rmse, df_metrics = verification_new(df_reviews_per_movies_dim, prepared, n)
        model_mae_list.append(mae)
        model_rmse_list.append(rmse)
        mean_mae = np.mean(mae)
        mean_rmse = np.mean(rmse)
        per_MeanMae_list.append(mean_mae)
        per_MeanRmse_list.append(mean_rmse)
    
    column_names = [f"col_{i}" for i in range(num_test_users)]
    df_model_mae = pd.DataFrame(model_mae_list, columns=column_names)
    df_model_rmse = pd.DataFrame(model_rmse_list, columns=column_names)
    df_model_mae.to_csv(f"result/model_name_nae.csv", index=False, encoding="utf-8-sig")
    df_model_rmse.to_csv(f"result/model_name_rmse.csv", index=False, encoding="utf-8-sig")
    df_csv = pd.DataFrame({"model_name":model_name_list,"MAE": per_MeanMae_list, "RMSE": per_MeanRmse_list})
    df_csv.to_csv(f"result/model_name.csv", index=False, encoding="utf-8-sig")

    stat_mae, stat_rmse, p_mae, p_rmse, max_mean_mae_index, second_max_mean_mae_index, max_mean_rmse_index, second_max_mean_rmse_index = maeAndRmse_wilcoxon(model_mae_list, model_rmse_list, per_MeanMae_list, per_MeanRmse_list)
    df_csv_wilcoxon = pd.DataFrame({"stat_mae":stat_mae,"stat_rmse": stat_rmse, "p_mae": p_mae,"p_rmse": p_rmse, "max_mean_mae_index": max_mean_mae_index, "second_max_mean_mae_index":second_max_mean_mae_index,"max_mean_rmse_index": max_mean_rmse_index, "second_max_mean_rmse_index": second_max_mean_rmse_index})
    df_csv_wilcoxon.to_csv(f"result/model_wilcoxon.csv", index=False, encoding="utf-8-sig")

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
#         num_evaluated_users,mae,rmse, df_metrics = verification(df_reviews_per_movies_dim, prepared)
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


if __name__ == "__main__":
    main()
