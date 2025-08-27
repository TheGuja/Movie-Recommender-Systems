import torch
import pickle

class Recommendations():
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device

        self.model = torch.load(self.model_path, weights_only=False, map_location=self.device)
        self.model.to(device)

        with open(r"C:\Users\jiag2\Coding\MovieLens\Matrix Factorization\config\user_id_mapping.pkl", "rb") as f:
            self.userID_to_mappedUserID = pickle.load(f)

        with open(r"C:\Users\jiag2\Coding\MovieLens\Matrix Factorization\config\reverse_movie_id_mapping", "rb") as f:
            self.mappedMovieID_to_movieID = pickle.load(f)

        with open(r"C:\Users\jiag2\Coding\MovieLens\Matrix Factorization\config\movieID_to_movieName_mapping.pkl", "rb") as f:
            self.movieID_to_movieName = pickle.load(f)

        with open(r"C:\Users\jiag2\Coding\MovieLens\Matrix Factorization\config\movieID_to_tmdbID_mapping.pkl", "rb") as f:
            self.movieID_to_tmdbID = pickle.load(f)

        self.numUsers = len(self.userID_to_mappedUserID)
        self.numMovies = len(self.mappedMovieID_to_movieID)
        
        # self.topMovieIDs = []
        

    # Recommends the top k movie ids
    def recommend_movie_IDs(self, user_id, topK=10):
        self.model.eval()

        mappedUserID = self.userID_to_mappedUserID[user_id]
        userTensor = torch.tensor([mappedUserID] * self.numMovies, device=self.device)
        movieTensor = torch.tensor(range(self.numMovies), device=self.device)

        with torch.no_grad():
            scores = self.model(userTensor, movieTensor)

        topScores, topMappedMovieIDs = torch.topk(scores, k=topK)
        topScores, topMappedMovieIDs = topScores.cpu().tolist(), topMappedMovieIDs.cpu().tolist()

        topMovieIDs = [int(self.mappedMovieID_to_movieID[movie]) for movie in topMappedMovieIDs]

        return topMovieIDs
    
    # Gets movie name from movie id
    def getMovieName(self, movieID):
        return self.movieID_to_movieName[movieID]
    
    # Gets tmdbID from movieID
    def gettmdbID(self, movieID):
        return self.movieID_to_tmdbID[movieID]