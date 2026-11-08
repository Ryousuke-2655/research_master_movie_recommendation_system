import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor 
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR 

import tensorflow as tf
from tensorflow.keras import layers, Model

from dimensionarity_reduction import dim_reduction_UMAP
from test_user_candidates import build_user_indices, prepare_user_splits_and_neighbors, user_candidates_selection

# ランダムフォレスト
def randomForest_model(X_train, X_valid, y_train, y_valid, X_test, y_test):
    # 列名を全て文字列に統一
    X_train.columns = X_train.columns.astype(str)
    X_valid.columns = X_valid.columns.astype(str)
    X_test.columns  = X_test.columns.astype(str)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse

# XGBoost
def xgboost_model(X_train, X_valid, y_train, y_valid, X_test, y_test):
    # 列名を全て文字列に統一
    X_train.columns = X_train.columns.astype(str)
    X_valid.columns = X_valid.columns.astype(str)
    X_test.columns  = X_test.columns.astype(str)
    model = xgb.XGBRegressor(random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)] , verbose=False)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse

# DNN→再コーテイングが必要
def mlp_regression_model(X_train, X_valid, y_train, y_valid, X_test, y_test):
    # 列名を全て文字列に統一
    X_train.columns = X_train.columns.astype(str)
    X_valid.columns = X_valid.columns.astype(str)
    X_test.columns  = X_test.columns.astype(str)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)
    X_test_s = scaler.transform(X_test)

    num_features = len(X_train.columns) 
    
    inputs = tf.keras.layers.Input(shape=(num_features,), name='inputs')

    wide_hidden_1 = tf.keras.layers.Dense(256, activation='relu', name='wide_relu_1')(inputs)
    wide_hidden_1_drop = tf.keras.layers.Dropout(0.5)(wide_hidden_1)
    wide_hidden_2 = tf.keras.layers.Dense(256, activation='relu', name='wide_relu_2')(wide_hidden_1_drop)
    wide_hidden_2_drop = tf.keras.layers.Dropout(0.5)(wide_hidden_2)
    wide_hidden = tf.keras.layers.Dropout(0.5)(wide_hidden_2_drop)

    deep_hidden_1 = tf.keras.layers.Dense(256, activation='relu', name='deep_relu_1')(inputs)
    deep_hidden_1_drop = tf.keras.layers.Dropout(0.5)(deep_hidden_1)
    deep_hidden_2 = tf.keras.layers.Dense(256, activation='relu', name='deep_relu_2')(deep_hidden_1_drop)
    deep_hidden = tf.keras.layers.Dropout(0.5)(deep_hidden_2)

    sync_hidden = layers.add([wide_hidden, deep_hidden], name="synchronization_layer")

    hidden_1 = tf.keras.layers.Dense(128, activation='relu', name='relu_1')(sync_hidden)
    hidden_1_drop = tf.keras.layers.Dropout(0.5)(hidden_1)
    hidden_2 = tf.keras.layers.Dense(64, activation='relu', name='relu_2')(hidden_1_drop)
    hidden_2_drop = tf.keras.layers.Dropout(0.5)(hidden_2)
    hidden_3 = tf.keras.layers.Dense(32, activation='relu', name='relu_3')(hidden_2_drop)
    hidden_3_drop = tf.keras.layers.Dropout(0.5)(hidden_3)
    outputs = tf.keras.layers.Dense(1, activation='linear', name='output')(hidden_3_drop)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='functional')
    model.compile(optimizer="sgd", loss="mse", metrics=["mae"])

    model.fit(X_train_s, y_train, epochs=100, batch_size=128, validation_data=(X_valid_s, y_valid))
    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse
    

# SVM
def support_vector_machine_model(X_train, X_valid, y_train, y_valid, X_test, y_test):
    # 列名を全て文字列に統一
    X_train.columns = X_train.columns.astype(str)
    X_valid.columns = X_valid.columns.astype(str)
    X_test.columns  = X_test.columns.astype(str)
    model = SVR(kernel="rbf")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse

# knn
def knearest_neighbors_model(X_train, X_valid, y_train, y_valid, X_test, y_test):
    # 列名を全て文字列に統一
    X_train.columns = X_train.columns.astype(str)
    X_valid.columns = X_valid.columns.astype(str)
    X_test.columns  = X_test.columns.astype(str)
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse

def model_select(name, X_train, X_valid, y_train, y_valid, X_test, y_test):
    if name == 'ran':
        mae,rmse =  randomForest_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
        return mae,rmse
    elif name == 'xg':
        mae,rmse =  xgboost_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
        return mae,rmse
    elif name == 'mlp':
        mae,rmse =  mlp_regression_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
        return mae,rmse
    elif name == 'svm':
        mae,rmse =  support_vector_machine_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
        return mae,rmse
    else:
        mae,rmse =  knearest_neighbors_model(X_train, X_valid, y_train, y_valid, X_test, y_test)
        return mae,rmse

def verification(df_reviews_per_movies, prepared, name):
    mae_list = []
    rmse_list = []
    
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

        mae, rmse = model_select(name, X_train, X_valid, y_train, y_valid, X_test, y_test)

        evaluated_users += 1

        mae_list.append(mae)
        rmse_list.append(rmse)

    df_metrics =pd.concat([pd.DataFrame(mae_list, columns = ['MAE']), pd.DataFrame(mae_list, columns = ['RMSE'])], axis = 1)
    
    return mae_list, rmse_list, df_metrics

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
    num_test_users = 100

    # ファイル読み込み
    df_reviews_per_movie = pd.read_csv(reviews_file_path)
    if os.path.exists(ratings_file_path):
        df_ratings = pd.read_json(ratings_file_path, lines=True)
    else:
        print(f"ファイルが見つかりません: {ratings_file_path}")
    
    df_re = df_reviews_per_movie.drop(columns=['item_id'])
    df_re_dimre =dim_reduction_UMAP(df_re) 
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
        mae,rmse, df_metrics = verification(df_reviews_per_movies_dim, prepared, n)
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
    df_csv_wilcoxon = pd.DataFrame({
        "stat_mae": [stat_mae],
        "stat_rmse": [stat_rmse],
        "p_mae": [p_mae],
        "p_rmse": [p_rmse],
        "max_mean_mae_index": [max_mean_mae_index],
        "second_max_mean_mae_index": [second_max_mean_mae_index],
        "max_mean_rmse_index": [max_mean_rmse_index],
        "second_max_mean_rmse_index": [second_max_mean_rmse_index],
    })
    df_csv_wilcoxon.to_csv(f"result/model_wilcoxon.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()