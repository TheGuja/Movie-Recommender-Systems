import pandas as pd
import os
from dotenv import load_dotenv

# Import and load datasets
load_dotenv()

ratings_filepath = os.getenv('RATINGS_FILEPATH')
movies_filepath = os.getenv('MOVIE_FILEPATH')

ratings = pd.read_csv(ratings_filepath, delimiter='::', engine='python')
movies = pd.read_csv(movies_filepath, delimiter='::', engine='python')

# Merge dataframes along MovieID column
merged_df = movies.merge(ratings, on='MovieID')
print(merged_df.head())

# Make a user movie matrix
userMovieMatrix = merged_df.pivot_table(index='UserID', columns='Title', values='Rating')
print(userMovieMatrix.head())

# Center the ratings to avoid user bias
userMovieMatrixCentered = userMovieMatrix.subtract(userMovieMatrix.mean(axis=0), axis='columns')

# Make a similarity matrix - adjusted cosine similarity
similarityMatrix = userMovieMatrixCentered.corr()
print(similarityMatrix.head())

# Define parameters
topK = 10
numRecommendations = 10
userID = 92

# Find what movies the user has watched
# pandas dataframe
hasWatched = userMovieMatrix[userMovieMatrix.index == userID].dropna(how='all', axis=1)

# Find what movies the user has not watched
colsToDrop = list(hasWatched.columns)
notWatched = userMovieMatrix.drop(labels=colsToDrop, axis=1)

# Calculate the predicted ratings for each movie
recommendedMovies = []
predictedRatings = []

for movieNotWatched in notWatched.columns:
    ratingSum = 0
    similaritySum = 0

    for movieWatched in hasWatched.columns:
        ratingWatched = userMovieMatrix[movieWatched][userID]

        similarity = similarityMatrix[movieWatched][movieNotWatched]
        if not pd.isna(ratingWatched):
            ratingSum += ratingWatched * similarity
            similaritySum += abs(similarity)
            
    if similaritySum > 0:
        predictedRating = ratingSum / similaritySum
        recommendedMovies.append(movieNotWatched)
        predictedRatings.append(predictedRating)

results = pd.DataFrame(list(zip(recommendedMovies, predictedRatings)), columns=['Movie', 'Predicted Rating']).sort_values('Predicted Rating', ascending=False).head(numRecommendations)
print(results)