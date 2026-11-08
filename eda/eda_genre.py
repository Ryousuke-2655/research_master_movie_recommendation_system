import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer

def main():
    file_path = "data/tmdb_movies_details_1960_2025.csv"

    df = pd.read_csv(file_path)
    df_new = df[["id", "genre_ids"]]
    df_new["genre_ids"] = df_new["genre_ids"].fillna("[]").apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()
    df_genre_onehot = pd.DataFrame(mlb.fit_transform(df_new["genre_ids"]), index =df_new.index, columns = [f"genre_{gid}" for gid in mlb.classes_])
    df_new = pd.concat([df_new,df_genre_onehot], axis=1)
    print(df_genre_onehot.columns.tolist())
    print(df_new.info())

if __name__ == "__main__":
    main()