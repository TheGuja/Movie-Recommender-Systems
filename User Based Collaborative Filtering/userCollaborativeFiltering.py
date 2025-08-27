import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
movieFilepath = os.getenv('MOVIE_FILEPATH')
ratingsFilepath = os.getenv('RATINGS_FILEPATH')

# Import datasets
movies = pd.read_csv(movieFilepath, delimiter='::', engine='python')
ratings = pd.read_csv(ratingsFilepath, delimiter='::', engine='python')


# Merge the datasets along MovieID column
merged_df = movies.merge(ratings, on='MovieID')