import torch
from torch.utils.data import DataLoader

from dataset import BurnoutDataset, ModelHistory
from model_factory import Model
from tqdm import tqdm


def train_model(model, batch_size, epochs, learning_rate, lr_decay, optimizer, criterion, device, metrics: dict = None):

    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=lr_decay)
    criterion = criterion()

    train_data = DataLoader(BurnoutDataset("train"), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(BurnoutDataset("val"), batch_size=batch_size, shuffle=True)

    if metrics is not None:
        history = ModelHistory(**metrics)
    else:
        history = ModelHistory()

    for epoch in range(epochs):

        model.train()
        pbar = tqdm(train_data)

        running_loss = 0
        current_epoch_preds = []
        current_epoch_targets = []

        for batch in pbar:
            x, y = batch
            x, y = x.to(device).float(), y.to(device)

            out = model(x)

            loss = criterion(out, y)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

            predicted = torch.argmax(out, dim=1)

            current_epoch_preds.extend(predicted.detach().cpu().tolist())
            current_epoch_targets.extend(y.detach().cpu().tolist())


        running_loss = running_loss / len(train_data)
        history.add_loss(running_loss)
        history.calculate_metrics(current_epoch_preds, current_epoch_targets)

    return model, history




