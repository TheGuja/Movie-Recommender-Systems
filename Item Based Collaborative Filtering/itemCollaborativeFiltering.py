import pandas as pd
import os
from dotenv import load_dotenv

# Import and load datasets
load_dotenv()

ratings_filepath = os.getenv('RATINGS_FILEPATH')
movies_filepath = os.getenv('MOVIE_FILEPATH')

ratings = pd.read_csv(ratings_filepath, delimiter='::', engine='python')
movies = pd.read_csv(movies_filepath, delimiter='::', engine='python')