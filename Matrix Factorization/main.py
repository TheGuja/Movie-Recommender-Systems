import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Dataset import MovieLensDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from MatrixFactorizationModel import MatrixFactorizationModel
from train_test_loop import training, testing

import os
from dotenv import load_dotenv

load_dotenv()
ratings_filepath = os.getenv('25ML_RATINGS_FILEPATH')

# Load dataset
df = pd.read_csv(ratings_filepath, delimiter=",", engine="python")
df = pd.read_csv()
df.drop("timestamp", axis=1, inplace=True)

# Get number of unique user and movie ids
uniqueUserIDs = df['userId'].unique()
uniqueMovieIDs = df['movieId'].unique()

# Map user and movie ids
mappingUserID = {user:index for index, user in enumerate(uniqueUserIDs)}
mappingMovieID = {movie:index for index, movie in enumerate(uniqueMovieIDs)}

num_users = len(mappingUserID)
num_movies = len(mappingMovieID)

# Apply mapping to dataframe
df['userId'] = df['userId'].map(mappingUserID)
df['movieId'] = df['movieId'].map(mappingMovieID)

# Split dataset
rest, test = train_test_split(df, test_size=0.2, shuffle=True, stratify=df['rating'])
train, val= train_test_split(rest, test_size=0.2, shuffle=True, stratify=rest['rating'])

# Load into custom dataset
train_dataset = MovieLensDataset(train)
val_dataset = MovieLensDataset(val)
test_dataset = MovieLensDataset(test)

# Make dataloaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=1024)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=1024)

# Define model
model = MatrixFactorizationModel(num_users=num_users, num_movies=num_movies, embedding_dim=64)

# Training Loop
train_loss_values, val_loss_values = training(model=model, train_loader=train_dataloader, val_loader=val_dataloader, optimizer=torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001), loss_fn=nn.MSELoss(), epochs=30)
testing(model=model, test_loader=test_dataloader, loss_fn=nn.MSELoss())