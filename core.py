# core.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

def compute_f1_score(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return f1_score(y_true, y_pred, average="macro")

def train(dataloader, model, loss_fn, optimizer, device):
    loss_epoch = torch.tensor(0.).to(device)
    model.train()
    y_true = []
    y_pred = []
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        # if y is a dict (common in detection tasks), move each tensor inside it
        if isinstance(y, dict):
            y = {k: v.to(device) for k, v in y.items()}
        else:
            y = y.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Compute prediction error
        pred = model(X)

        # Compute loss
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        loss_epoch += loss.detach()

        # Apply sigmoid to output for binary/multi-label classification and threshold at 0.5
        pred = (torch.sigmoid(pred) > 0.5).float()

        # Store true and predicted values for F1 score computation
        y_true.append(y.cpu())
        y_pred.append(pred.cpu())

    # Concatenate all true and predicted labels
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    # Compute F1 score
    f1 = compute_f1_score(y_true, y_pred)

    loss_epoch = (loss_epoch / num_batches).cpu().item()

    return {"train_loss": loss_epoch, "f1_score": f1}

def val(dataloader, model, loss_fn, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = torch.tensor(0.).to(device)
    y_true = []
    y_pred = []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Get model predictions
        pred = model(X)

        # Compute loss
        val_loss += loss_fn(pred, y)

        # Apply sigmoid for multi-label classification and threshold at 0.5
        pred = (torch.sigmoid(pred) > 0.5).float()

        # Store true and predicted values for F1 score computation
        y_true.append(y.cpu())
        y_pred.append(pred.cpu())

    # Concatenate all true and predicted labels
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)

    # Compute F1 score 
    f1 = compute_f1_score(y_true, y_pred)

    val_loss = (val_loss / num_batches).cpu().item()

    return {"val_loss": val_loss, "f1_score": f1}
