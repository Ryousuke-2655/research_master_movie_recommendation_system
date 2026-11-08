import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import random

#コサイン類似度
def CosineSimilarity(user_data, user_ratings_map, user_rows_map, item_to_users, top_k=15):
    user_ratings = user_data.groupby("movie_id")["rating"].mean()
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
    df_ratings["rating_count"]=df_ratings.groupby("author")["rating"].transform("count")
    df_filter_user = df_ratings[(df_ratings["rating_count"] >= 50) & (df_ratings["rating_count"] <= 100)].reset_index(drop=True)
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

    for uid, group in other_user_data.groupby("author"):
        user_rows_map[uid] = group
        ratings = group.groupby("movie_id")["rating"].mean()
        user_ratings_map[uid] = ratings
        for item_id in ratings.index:
            item_to_users.setdefault(item_id, []).append(uid)
        
    return user_ratings_map, user_rows_map, item_to_users

#  データ分割関数（類似ユーザデータ（train）,検証ユーザデータ（valid,test））
def prepare_user_splits_and_neighbors(test_users, ratings_by_user, division_rate,user_ratings_map, user_rows_map, item_to_users):
    prepared = {}
    for author in test_users:
        target_user_data = ratings_by_user[author]
        train, temp = train_test_split(target_user_data, test_size=division_rate, random_state=42)
        valid, test = train_test_split(temp, test_size= 2/3, random_state=42)
        neighbors = CosineSimilarity(train, user_ratings_map, user_rows_map, item_to_users)
        prepared[author] = {"valid": valid, "test": test, "neighbors": neighbors}
    return prepared

#  データ分割関数（類似ユーザデータ（train）,検証ユーザデータ（valid,test））
def prepare_user_splits_mf(test_users, ratings_by_user, division_rate,user_ratings_map, user_rows_map, item_to_users):
    prepared = {}
    for author in test_users:
        target_user_data = ratings_by_user[author]
        train, temp = train_test_split(target_user_data, test_size=division_rate, random_state=42)
        valid, test = train_test_split(temp, test_size= 2/3, random_state=42)
        neighbors = CosineSimilarity(train, user_ratings_map, user_rows_map, item_to_users)
        prepared[author] = {"valid": valid, "test": test, "neighbors": neighbors}
    return prepared

#  データ分割関数（類似ユーザデータ（train）,検証ユーザデータ（valid,test））
def prepare_user_splits_and_neighbors_onePerson(test_users, ratings_by_user, division_rate,user_ratings_map, user_rows_map, item_to_users):
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
    df_train =train[["movie_id","title","rating"]]
    return prepared ,df_train


#  データ分割関数（類似ユーザデータ（train）,検証ユーザデータ（valid,test））
def prepare_user_splits_and_neighbors_recommendation(test_users, ratings_by_user, division_rate,user_ratings_map, user_rows_map, item_to_users):
    prepared = {}
    for author in test_users:
        target_user_data = ratings_by_user[author]
        train, valid = train_test_split(
            target_user_data, test_size=division_rate, random_state=42
        )
        neighbors = CosineSimilarity(
            train, user_ratings_map, user_rows_map, item_to_users
        )

        prepared[author] = {
            "valid": valid,
            "neighbors": neighbors,
        }
    df_train =train[["movie_id","title","rating"]]
    return prepared ,df_train



