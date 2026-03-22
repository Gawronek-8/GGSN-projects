import torch
from torch.utils.data import DataLoader

from dataset import BurnoutDataset, ModelHistory
from tqdm import tqdm


def _train_one_epoch(model, train_data, device, criterion, optimizer, history = None):

        model.train()

        current_epoch_preds = []
        current_epoch_targets = []
        epoch_loss = 0
        pbar = tqdm(train_data)

        for batch in pbar:
            x, y = batch
            x, y = x.to(device).float(), y.to(device)

            out = model(x)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

            predicted = torch.argmax(out, dim=1)

            current_epoch_preds.extend(predicted.detach().cpu().tolist())
            current_epoch_targets.extend(y.detach().cpu().tolist())

        if history is None:
            return


        epoch_loss = epoch_loss / len(train_data)
        history.add_loss("train", epoch_loss)
        history.calculate_metrics("train", current_epoch_targets, current_epoch_preds)


def _val_one_epoch(model, val_data, device, criterion, history = None):
    current_val_preds = []
    current_val_targets = []
    val_loss = 0

    pbar = tqdm(val_data)

    model.eval()

    with torch.no_grad():
        for batch in pbar:
            x, y = batch
            x, y = x.to(device).float(), y.to(device)

            out = model(x)

            loss = criterion(out, y)

            val_loss += loss.item()

            predicted = torch.argmax(out, dim=1)

            current_val_preds.extend(predicted.detach().cpu().tolist())
            current_val_targets.extend(y.detach().cpu().tolist())


    if history is None:
        return

    val_loss = val_loss / len(val_data)
    history.add_loss("val", val_loss)
    history.calculate_metrics("val", current_val_targets, current_val_preds)



def train_model(model, batch_size, epochs, learning_rate, weight_decay, optimizer, criterion, device, metrics: dict = None):

    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_data = DataLoader(BurnoutDataset("train"), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(BurnoutDataset("val"), batch_size=batch_size, shuffle=True)

    if metrics is not None:
        history = ModelHistory(["train", "val"], model.name,  **metrics)
    else:
        history = ModelHistory(["train", "val"], model.name)

    for epoch in range(epochs):

        _train_one_epoch(model, train_data, device, criterion, optimizer, history)
        _val_one_epoch(model, val_data, device, criterion, history)


    train_info = {
        "epochs": epochs,
        "optimizer" : optimizer.__class__.__name__,
        "criterion" : criterion.__class__.__name__,
        "batch_size" : batch_size,
        "learning_rate" : learning_rate,
        "lr_decay" : weight_decay,
    }

    model.add_training_info(train_info)


    return model, history
