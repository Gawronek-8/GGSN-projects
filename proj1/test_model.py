from tqdm import tqdm
import torch

def test_model(model, test_data, criterion, device):
    current_val_preds = []
    current_val_targets = []
    val_loss = 0

    pbar = tqdm(test_data)

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

    return val_loss, current_val_preds, current_val_targets
