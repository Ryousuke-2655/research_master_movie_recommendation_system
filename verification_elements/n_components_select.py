import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from sklearn.model_selection import train_test_split
import random

#コサイン類似度
def CosineSimilarity(user_data, user_ratings_map, user_rows_map, item_to_users, top_k=20):
    user_ratings = user_data.groupby('item_id')['rating'].mean()
    user_mean = user_ratings.mean()
    user_centered = user_ratings - user_mean

    user_item_set = set(user_ratings.index)

    candidate_uids = set()
    for item_id in user_item_set:
        candidate_uids.update(item_to_users.get(item_id, ()))

    similarities = []

    for uid in candidate_uids:
        other_ratings = user_ratings_map[uid]
        common_items = user_item_set.intersection(other_ratings.index)
        if not common_items:
            continue

        common = list(common_items)
        other_mean = other_ratings.mean()

        t = user_centered[common].loc[common].to_numpy()
        o = (other_ratings.loc[common]  - other_mean).to_numpy()
        denom = np.linalg.norm(t) * np.linalg.norm(o)
        sim = float(np.dot(t, o) / denom) if denom != 0 else 0.0

        similarities.append((sim, uid))
    
    if not similarities:
        return pd.DataFrame()

    similarities.sort(key=lambda x: x[0], reverse=True)
    top_users = []

    for sim_and_uid in similarities[:top_k]:
        uid = sim_and_uid[1]
        top_users.append(uid)

    neighbor_rows = []

    for uid in top_users:
        neighbor_rows.append(user_rows_map[uid])

    similar_user_data = pd.concat(neighbor_rows, ignore_index=True)
    return similar_user_data


# テストユーザ選択
def user_candidates_selection(df_ratings,num_test_users):
    df_ratings['rating_count']=df_ratings.groupby('user_id')['rating'].transform('count')
    df_filter_user = df_ratings[(df_ratings['rating_count'] >= 50) & (df_ratings['rating_count'] <= 100)].reset_index(drop=True)
    test_user_candidates = df_filter_user['user_id'].unique()
    test_user_candidates_list = list(test_user_candidates)

    if len(test_user_candidates_list) < num_test_users:
        raise ValueError(f"Not enough test user candidates to sample {num_test_users} users from")

    test_users = random.sample(test_user_candidates_list, num_test_users)
    
    return test_users

# 類似度用の前処理: ユーザ別の行・平均評価・映画→ユーザの転置を1回だけ作る
def build_user_indices(other_user_data):
    user_ratings_map = {}
    user_rows_map = {}
    item_to_users = {}

    for uid, group in other_user_data.groupby('user_id'):
        user_rows_map[uid] = group
        ratings = group.groupby('item_id')['rating'].mean()
        user_ratings_map[uid] = ratings
        for item_id in ratings.index:
            item_to_users.setdefault(item_id, []).append(uid)
    return user_ratings_map, user_rows_map, item_to_users

#  データ分割関数（類似ユーザデータ（train）,検証ユーザデータ（valid,test））
def prepare_user_splits_and_neighbors(test_users, ratings_by_user, division_rate,user_ratings_map, user_rows_map, item_to_users):
    prepared = {}
    for author in test_users:
        target_user_data = ratings_by_user[author]
        train, temp = train_test_split(
            target_user_data, test_size=division_rate, random_state=42
        )
        valid, test = train_test_split(temp, test_size= 2/3, random_state=42)
        neighbors = CosineSimilarity(
            train, user_ratings_map, user_rows_map, item_to_users
        )
        prepared[author] = {
            "valid": valid,
            "test": test,
            "neighbors": neighbors,
        }
    return prepared

def verification(df_reviews_per_movies, prepared):
    mae_list = []
    rmse_list = []
    evaluated_users = 0  # テストユーザ数
    
    for author, pack in prepared.items():
        target_user_data_valid_me = pack['valid'].merge(df_reviews_per_movies, on='item_id', how='inner')
        target_user_data_test_me = pack['test'].merge(df_reviews_per_movies, on='item_id', how='inner')
        similarity_user_data_me = pack['neighbors'].merge(df_reviews_per_movies, on='item_id', how='inner')
        X_train = similarity_user_data_me.drop(columns=['rating','user_id', 'item_id'])
        X_valid = target_user_data_valid_me.drop(columns=['rating','item_id'])
        X_test = target_user_data_test_me.drop(columns=['rating','item_id'])

        y_train = similarity_user_data_me['rating']
        y_valid = target_user_data_valid_me['rating']
        y_test = target_user_data_test_me['rating']

        mae,rmse =  randomForest_model(X_train, X_valid, y_train, y_valid, X_test, y_test)

        mae_list.append(mae)
        rmse_list.append(rmse)

    df_metrics =pd.concat([pd.DataFrame(mae_list, columns = ['MAE']), pd.DataFrame(mae_list, columns = ['RMSE'])], axis = 1)
    
    return evaluated_users, mae_list, rmse_list, df_metrics

def main():
    # ファイルパス
    ratings_file_path = 'data/updated_ratings.json'
    reviews_file_path = 'data/review_mean_embeddings_per_movie.csv'

    per_MeanMae_list = []
    per_MeanRmse_list = []
    n_components  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380]
    division_rate =0.3
    num_test_users = 100

    # ファイル読み込み
    df_reviews_per_movie = pd.read_csv(reviews_file_path)
    if os.path.exists(ratings_file_path):
        df_ratings = pd.read_json(ratings_file_path, lines=True)
    else:
        print(f"ファイルが見つかりません: {ratings_file_path}")
    
    df_re = df_reviews_per_movie.drop(columns=['item_id'])
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

    for i,n in enumerate(n_components):
        print(n)
        df_re_dimre = dim_reduction_UMAP(df_re, n)
        df_notre=pd.DataFrame(notre)
        df_re_dimre_new =pd.DataFrame(df_re_dimre)
        df_reviews_per_movies_dim = pd.concat([df_notre, df_re_dimre_new], axis=1)
        num_evaluated_users,mae,rmse, df_metrics = verification(df_reviews_per_movies_dim, prepared)
        mean_mae = np.mean(mae)
        mean_rmse = np.mean(rmse)
        per_MeanMae_list.append(mean_mae)
        per_MeanRmse_list.append(mean_rmse)
        if i ==0:
            print('クリア')
    
    df_csv = pd.DataFrame({"n_components":n_components,"MAE": per_MeanMae_list, "RMSE": per_MeanRmse_list})
    
    df_csv.to_csv(f"result/n_components.csv", index=False, encoding="utf-8-sig")
    
    plt.plot(n_components, per_MeanMae_list, marker="o")
    plt.title('n_componentsごとのMAE')
    plt.xlabel('n_components')
    plt.ylabel('MAE')
    plt.xlim(0, None) 
    plt.ylim(0, 2) 
    plt.savefig('result/mae_n_components.png')
    plt.show()

    plt.plot(n_components, per_MeanRmse_list, marker="o")
    plt.title('n_componentsごとのRMSE')
    plt.xlabel('n_components')
    plt.ylabel('RMSE')
    plt.xlim(0, None) 
    plt.ylim(0, 2) 
    plt.savefig('result/rmse_n_components.png')
    plt.show()


if __name__ == "__main__":
    main()