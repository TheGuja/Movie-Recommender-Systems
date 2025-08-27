import torch
import torch.nn as nn

class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=self.embedding_dim)
        self.movie_embedding = nn.Embedding(num_embeddings=num_movies, embedding_dim=self.embedding_dim)

        # Bias
        self.user_bias = nn.Embedding(num_embeddings=num_users, embedding_dim=1)
        self.movie_bias = nn.Embedding(num_embeddings=num_movies, embedding_dim=1)

    def forward(self, user, movie):
        user_vec = self.user_embedding(user)
        movie_vec = self.movie_embedding(movie)
        dot = (user_vec * movie_vec).sum(1)
        bias = self.user_bias(user).squeeze() + self.movie_bias(movie).squeeze()

        return dot + bias