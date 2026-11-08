import os
import json
import requests
import csv

from dotenv import load_dotenv
from time import sleep

class TMDB:
    def __init__(self, token):
        self.token = token
        self.headers_ = {'Authorization': f'Bearer {self.token}'}  
        self.base_url_ = 'https://api.themoviedb.org/3'
        self.img_base_url_ = 'https://image.tmdb.org/t/p/w500'

    def _json_by_get_request(self, url, params={}):
        try:
            res = requests.get(url, headers=self.headers_, params=params)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print("エラー : ",e)
            return None
    
    def get_review_count(self, movie_id, language="en-US"):
        url = f"{self.base_url_}/movie/{movie_id}/reviews"
        params = {"language": language, "page": 1}
        data = self._json_by_get_request(url, params)
        if data is None:
            return 0
        return data.get("total_results", 0)
    
    def get_movie_by_year(self, year, min_reviews = 2 ,language= "en-US"):
        movies = []
        page = 1
        total_pages = 1

        while page <= total_pages:
            url = f'{self.base_url_}/discover/movie'
            params = {"primary_release_year": year, "page": page, "language": language, "include_adult": False, "sort_by": "popularity.desc",}
            data = self._json_by_get_request(url, params=params)

            if data is None:
                break
            
            total_pages = min(data.get("total_pages", 1), 500)

            for movie in data.get("results", []):
                movie_id = movie.get("id")
                if movie_id is None:
                    continue
                review_count = self.get_review_count(movie_id, language=language)
                sleep(0.25)

                if review_count >= min_reviews:
                    movie["review_count"] = review_count
                    movies.append(movie)
            
            page += 1
            sleep(0.25)

        return movies
    
    def get_movie_reviews(self, movie_id):
        url = f'{self.base_url_}/movie/{movie_id}/reviews'
        return self._json_by_get_request(url)

    def get_all_reviews(self, movie_id, language="en-US"):
        reviews = []
        page = 1
        total_pages = 1

        while page <= total_pages:
            url = f"{self.base_url_}/movie/{movie_id}/reviews"
            params = {"language": language, "page": page}
            data = self._json_by_get_request(url, params)
            if data is None:
                break

            total_pages = data.get("total_pages", 1)
            reviews.extend(data.get("results", []))
            page += 1
            sleep(0.25)

        return reviews

    def get_movie_credits(self, movie_id):
        url = f'{self.base_url_}/movie/{movie_id}/credits'
        return self._json_by_get_request(url)


def flatten_for_csv(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def load_existing_ids(file_name, id_column):
    existing_ids = set()
    if not os.path.exists(file_name):
        return existing_ids

    with open(file_name, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            value = row.get(id_column)
            if value:
                existing_ids.add(str(value))
    return existing_ids


def append_csv_rows(output_csv, rows, fieldnames, write_header):
    with open(output_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return False


def resolve_fieldnames(csv_path, row):
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                return list(reader.fieldnames)
    return list(row.keys())


def main():
    load_dotenv()
    token = os.getenv("API_TOKEN")
    api = TMDB(token)

    year = 1961  # 実行のたびにこの年だけ変える

    details_csv = "tmdb_movies_details.csv"
    reviews_csv = "tmdb_movies_reviews.csv"

    existing_movie_ids = load_existing_ids(details_csv, "id")
    write_details_header = not os.path.exists(details_csv)
    write_reviews_header = not os.path.exists(reviews_csv)

    details_fieldnames = None
    reviews_fieldnames = None

    print(f"==== {year}年を取得 ====")
    movies = api.get_movie_by_year(year)

    for movie in movies:
        movie_id = str(movie["id"])
        if movie_id in existing_movie_ids:
            continue

        credits = api.get_movie_credits(movie_id)
        sleep(0.5)
        cast = credits.get("cast", []) if credits else []
        crew = credits.get("crew", []) if credits else []

        detail_row = {k: flatten_for_csv(v) for k, v in movie.items()}
        detail_row["year"] = year
        detail_row["cast"] = flatten_for_csv(cast)
        detail_row["crew"] = flatten_for_csv(crew)

        if details_fieldnames is None:
            details_fieldnames = resolve_fieldnames(details_csv, detail_row)

        write_details_header = append_csv_rows(
            details_csv, [detail_row], details_fieldnames, write_details_header
        )
        existing_movie_ids.add(movie_id)

        review_results = api.get_all_reviews(movie_id)
        review_rows = []
        for review in review_results:
            row = {"movie_id": movie_id, "year": year}
            row.update({k: flatten_for_csv(v) for k, v in review.items()})
            review_rows.append(row)

        if not review_rows:
            continue

        if reviews_fieldnames is None:
            reviews_fieldnames = resolve_fieldnames(reviews_csv, review_rows[0])

        write_reviews_header = append_csv_rows(
            reviews_csv, review_rows, reviews_fieldnames, write_reviews_header
        )

    print(f"{year}年の追記が完了しました")


if __name__ == "__main__":
    main()
