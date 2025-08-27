from tqdm import tqdm
import torch

def training(model, train_loader, val_loader, optimizer, loss_fn, epochs, device="cuda"):
    model.to(device)

    train_loss_items = []
    val_loss_items = []
    for epoch in range(epochs):
        model.train()
        total_loss_train, total_loss_val = 0, 0

        for user_train, movie_train, rating_train in tqdm(train_loader):
            user_train = user_train.to(device)
            movie_train = movie_train.to(device)
            rating_train = rating_train.to(device)

            pred_train = model(user_train, movie_train)
            loss_train = loss_fn(pred_train, rating_train)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            total_loss_train += loss_train.item()

        model.eval()
        with torch.no_grad():
            for user_val, movie_val, rating_val in tqdm(val_loader):
                user_val = user_val.to(device)
                movie_val = movie_val.to(device)
                rating_val = rating_val.to(device)

                pred_val = model(user_val, movie_val)
                loss_val = loss_fn(pred_val, rating_val)

                total_loss_val += loss_val.item()
        
        train_loss = total_loss_train / len(train_loader)
        val_loss = total_loss_val / len(val_loader)
        
        print(f"Epoch {epoch + 1}/{epochs}: {train_loss}")
        print(f"Validation Loss: {val_loss}")

        train_loss_items.append(train_loss)
        val_loss_items.append(val_loss)

    return train_loss_items, val_loss_items

def testing(model, test_loader, loss_fn, device="cuda"):
    model.to(device)

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for user, movie, rating in test_loader:
            user = user.to(device)
            movie = movie.to(device)
            rating = rating.to(device)

            pred = model(user, movie)
            loss = loss_fn(pred, rating)

            total_loss += loss.item()

        print(f"Loss: {total_loss / len(test_loader)}")