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
# print(merged_df.head())

# Make user movie matrix
userMovieMatrix = merged_df.pivot_table(index='UserID', columns='Title', values='Rating')
# print(userMovieMatrix)

# Make a similarity matrix
similarityMatrix = userMovieMatrix.T.corr('pearson')
# print(similarityMatrix)

# Define the parameters
topK = 10
numRecommendations = 10
similarityThreshold = 0.4
userID = 100

# Find k-nearest neighbors to userID
kNN = similarityMatrix[similarityMatrix[userID] > similarityThreshold][userID].sort_values(ascending=False)[:topK]
print(kNN)