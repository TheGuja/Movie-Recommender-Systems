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

# Make a similarity matrix
similarityMatrix = userMovieMatrix.T.corr('pearson')

# Define the parameters
topK = 10
numRecommendations = 10
similarityThreshold = 0.4
userID = 100

# Find k-nearest neighbors to userID
kNN = similarityMatrix[similarityMatrix[userID] > similarityThreshold][userID].sort_values(ascending=False)[:topK]

# Find the movies that the user has watched
# This is a dataframe
hasWatched = userMovieMatrix[userMovieMatrix.index == userID].dropna(axis=1, how='all')

# Find the movies that the user has not watched
# This is also a dataframe
notWatched = userMovieMatrix[userMovieMatrix.index.isin(kNN.index)].dropna(axis=1, how='all')
notWatched.drop(labels=hasWatched.columns, axis=1, inplace=True, errors='ignore')

notWatchedMovies = notWatched.columns
recommendedMovies = []
predictedRatings = []
numRecommendations = 10

# Calculate the mean rating for userID
mu_u = userMovieMatrix[userMovieMatrix.index == userID].T.mean()[userID]

# Find movies to recommend based on the k-nearest neighbors
for movie in notWatchedMovies:
    ratingSum = 0
    similaritySum = 0

    for user in notWatched.index:
        rating = notWatched.loc[user][movie]
        similarityScore = similarityMatrix.loc[userID][user]

        if not pd.isna(rating):
            mu_v = userMovieMatrix[userMovieMatrix.index == user].T.mean()[user]
            meanCenteredRating = rating - mu_v
            ratingSum = meanCenteredRating * similarityScore
            similaritySum += ratingSum

    predictedRating = mu_u + (ratingSum / similaritySum)
    recommendedMovies.append(movie)
    predictedRatings.append(predictedRating)


results = pd.DataFrame(list(zip(recommendedMovies, predictedRatings)), columns=['Movie', 'Predicted Rating']).sort_values('Predicted Rating', ascending=False)[:numRecommendations]
print(results)