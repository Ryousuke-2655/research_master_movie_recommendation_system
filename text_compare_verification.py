import pandas as pd
import numpy as np

from text_embedding import sentence_embedding, mean_text_embedding_per_movie
from dimensionarity_reduction import dim_reduction_UMAP
from test_user_candidates import build_user_indices, prepare_user_splits_and_neighbors
from model import randomForest_model
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def genre_onehot(df):
    df["genre_ids"] = df["genre_ids"].fillna("[]").apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()
    df_genre_onehot = pd.DataFrame(mlb.fit_transform(df["genre_ids"]), index =df.index, columns = [f"genre_{gid}" for gid in mlb.classes_])
    df = pd.concat([df,df_genre_onehot], axis=1)
    df_new = df.drop(columns=["genre_ids"])
    return df_new

def verification(df, prepared, num_test_users):
    mae_list = []
    rmse_list = []
    author_list = []

    for author, pack in prepared.items():
        target_user_data_valid_me = pack["valid"].merge(df, on="movie_id", how="inner")
        target_user_data_test_me = pack["test"].merge(df, on="movie_id", how="inner")
        similarity_user_data_me = pack["neighbors"].merge(df, on="movie_id", how="inner")
        X_train = similarity_user_data_me.drop(columns=["rating","author", "movie_id"])
        X_valid = target_user_data_valid_me.drop(columns=["rating","movie_id"])
        X_test = target_user_data_test_me.drop(columns=["rating","movie_id"])

        y_train = similarity_user_data_me["rating"]
        y_valid = target_user_data_valid_me["rating"]
        y_test = target_user_data_test_me["rating"]

        mae,rmse = randomForest_model(X_train, X_valid, y_train, y_valid, X_test, y_test)

        mae_list.append(mae)
        rmse_list.append(rmse)

    for i in range(num_test_users):  
        name = f"user_{i}" 
        author_list.append(name)

    df_metrics =pd.concat([pd.DataFrame(author_list, columns = ["author_list"]), pd.DataFrame(mae_list, columns = ["MAE_llm"]), pd.DataFrame(mae_list, columns = ["RMSE_llm"])], axis = 1)
    # df_metrics =pd.concat([pd.DataFrame(author_list, columns = ["author_list"]), pd.DataFrame(mae_list, columns = ["MAE_overview"]), pd.DataFrame(mae_list, columns = ["RMSE_overview"])], axis = 1)
    # df_metrics =pd.concat([pd.DataFrame(author_list, columns = ["author_list"]), pd.DataFrame(mae_list, columns = ["MAE_review"]), pd.DataFrame(mae_list, columns = ["RMSE_review"])], axis = 1)
    # df_metrics =pd.concat([pd.DataFrame(author_list, columns = ["author_list"]), pd.DataFrame(mae_list, columns = ["MAE_genre"]), pd.DataFrame(mae_list, columns = ["RMSE_genre"])], axis = 1)
    
    return mae_list, rmse_list, df_metrics

def main():
    # ファイルパス
    target_users_file_path = "data/verification_users_data.csv"
    details_file_path = "data/tmdb_movies_details_1960_2025.csv"
    reviews_file_path = "data/candidate_users_data.csv"
    llm_text_file_path = "data/generate_text_qwen_100.csv"

    # 初期値
    division_rate =0.3
    num_test_users = 20

    # ファイル読み込み
    #検証ユーザ
    df_target_users = pd.read_csv(target_users_file_path)
    df_ratings_target = df_target_users[["author", "movie_id", "rating"]]
    test_user_unique = df_ratings_target["author"].unique()
    test_users = list(test_user_unique)

    df_details = pd.read_csv(details_file_path)
    df_details = df_details.rename(columns={"id": "movie_id"})
    df_reviews = pd.read_csv(reviews_file_path)
    df_ratings_candidate = df_reviews[["author", "movie_id", "rating"]]
    df_llm_text = pd.read_csv(llm_text_file_path)
    # df_overview_text = df_details [["movie_id","overview"]]
    # df_review_text = df_reviews[["movie_id","content"]]
    # df_genre = df_details [["movie_id","genre_ids"]]

    # df_overview_embedding, col_number = sentence_embedding(df_overview_text, "overview")

    # df_review_embedding, col_number = sentence_embedding(df_review_text, "content")
    # df_review_embedding_per_movie = mean_text_embedding_per_movie(df_review_embedding, col_number)

    df_llm_embedding, col_number = sentence_embedding(df_llm_text, "generate_text")

    # df_genre_onehot = genre_onehot(df_genre)
    
    # df_re = df_overview_embedding.drop(columns=["movie_id"])
    # df_re = df_review_embedding_per_movie.drop(columns=["movie_id"])
    df_re = df_llm_embedding.drop(columns=["movie_id"])

    df_re_dimre =dim_reduction_UMAP(df_re) 
    # notre = df_overview_embedding["movie_id"]
    # notre = df_review_embedding_per_movie["movie_id"]
    notre = df_llm_embedding["movie_id"]
    
    user_ratings_map, user_rows_map, item_to_users = build_user_indices(df_ratings_candidate)
    ratings_by_user = {}  

    for uid, g in df_ratings_target.groupby("author"): 
        not_uid = g.drop(columns=["author"]) 
        ratings_by_user[uid] = not_uid       

    prepared = prepare_user_splits_and_neighbors(test_users, ratings_by_user, division_rate, user_ratings_map, user_rows_map, item_to_users)

    df_notre=pd.DataFrame(notre)
    df_re_dimre_new =pd.DataFrame(df_re_dimre)
    df_reviews_per_movies_dim = pd.concat([df_notre, df_re_dimre_new], axis=1)
    mae, rmse, df_metrics = verification(df_reviews_per_movies_dim, prepared, num_test_users)
    # mae, rmse, df_metrics = verification(df_genre_onehot, prepared, num_test_users)
    mean_mae = np.mean(mae)
    mean_rmse = np.mean(rmse)

    df_metrics.to_csv(f"result/result_per_author_llm_qwen.csv", index=False, encoding="utf-8-sig")
    df_csv = pd.DataFrame({"MAE_llm": [mean_mae], "RMSE_llm": [mean_rmse]})
    df_csv.to_csv(f"result/result_llm_qwen.csv", index=False, encoding="utf-8-sig")

    # df_metrics.to_csv(f"result/result_per_author_overview.csv", index=False, encoding="utf-8-sig")
    # df_csv = pd.DataFrame({"MAE_overview": [mean_mae], "RMSE_overview": [mean_rmse]})
    # df_csv.to_csv(f"result/result_overview.csv", index=False, encoding="utf-8-sig")

    # df_metrics.to_csv(f"result/result_per_author_review.csv", index=False, encoding="utf-8-sig")
    # df_csv = pd.DataFrame({"MAE_review": [mean_mae], "RMSE_review": [mean_rmse]})
    # df_csv.to_csv(f"result/result_review.csv", index=False, encoding="utf-8-sig")

    # df_metrics.to_csv(f"result/result_per_author_genre.csv", index=False, encoding="utf-8-sig")
    # df_csv = pd.DataFrame({"MAE_genre": [mean_mae], "RMSE_genre": [mean_rmse]})
    # df_csv.to_csv(f"result/result_genre.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()