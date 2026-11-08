import pandas as pd
import numpy as np

details_file_path = "data/tmdb_movies_details_1960_2025.csv"
df_details = pd.read_csv(details_file_path)
print(df_details.info())