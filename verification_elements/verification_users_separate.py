import pandas as pd
import numpy as np
import random

def main():
    reviews_file_path = 'data/tmdb_movies_reviews_1960_2025.csv'
    df_reviews = pd.read_csv(reviews_file_path)

    df_reviews['rating_count']=df_reviews.groupby('author')['rating'].transform('count')
    df_filter_user = df_reviews[(df_reviews['rating_count'] >= 30) & (df_reviews['rating_count'] <= 60)].reset_index(drop=True)
    test_user_candidates = df_filter_user['author'].unique()

    test_user_candidates_list = list(test_user_candidates)
    test_users = random.sample(test_user_candidates_list, 20)
    exclude_users = set(test_users)
    target_user_data = df_reviews[
        df_reviews['author'].isin(exclude_users)
    ]
    other_user_data = df_reviews[
        ~df_reviews['author'].isin(exclude_users)
    ]

    target_user_data.to_csv(f"data/verification_users_data.csv", index=False, encoding="utf-8-sig")
    other_user_data.to_csv(f"data/candidate_users_data.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()