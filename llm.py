
import os
from unittest import result
import ollama
import json
import pandas as pd

def main():
    file_path = "data/llm_movie_information_data_1.json"
    output_file_path = "data/generate_text.csv"

    system_rule = """You will serve as an assistant to help me summarize which types of users would enjoy a specific movie.
    I will provide you with the title and an overview and a content_{number} of the movie. Here are the instructions:
    1. Iwill provide you with information in the form of a JSON file that describes the movie:
    movie_id:"ID for each movie"{"title":"the title of the movie", (if there is no title, I will set this value to "None")
    "overview":"a overview of the movie", (if there is no overview, Iwill set this value to "None")
    "content_{number}:"a review of the movie written by users", (if there is no overview, Iwill set this value to "None")}

    Requirements:
    1. Please provide your answer in JSON format, following this structure:
    {"movie_id":"ID for each movie(Refer to the input data)"
    "generate_text":"a summary of the movie based on the input data(title,overview,content)", (if there is no overview, Iwill set this value to "None")}
    2. Please ensure that the "generate_text" is no longer than 200 words.
    3. Do not provide any other text outside the JSON string."""

    with open(file_path, "r", encoding="utf-8-sig") as f:
        movies = json.load(f)
    
    if os.path.exists(output_file_path):
        df_exists = pd.read_csv(output_file_path,dtype={"movie_id":str})
        exists_ids = set(df_exists["movie_id"])
    else:
        exists_ids = set()
    for movie_id, info in movies.items():
        movie_id = str(movie_id)
        if movie_id in exists_ids:
            continue
        input_data = {"movie_id": movie_id, **info} 
        user_content = json.dumps(input_data, ensure_ascii=False)

        try:
            response = ollama.generate(
                model="gemma4",
                system = system_rule,
                prompt= user_content, 
                format = "json", 
            )
            result = json.loads(response.response)
        except Exception as e:
            print(f"失敗:movie_id={movie_id}({e})")
            continue
        row = {
            "movie_id":movie_id,
            "generate_text":result.get("generate_text","")
        }
        df = pd.DataFrame([row])
        write_header = not os.path.exists(output_file_path)
        df.to_csv(output_file_path,mode = "a",header = write_header, index = False, encoding="utf-8-sig")
        exists_ids.add(movie_id)
    print("完了")

if __name__ == "__main__":
    main()