from functorch.dim import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric
from tqdm import tqdm
import math
from dataset import FireDatasetForDetection, ModelHistory
import torch
from utils import collate_fn, is_better_value
from torchmetrics.detection.mean_ap import  MeanAveragePrecision
from config import MODEL_DIR
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor


def _train_one_epoch(model, train_data, device, optimizer):
    model.train()
    pbar = tqdm(train_data)

    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if device != "cuda":
            loss_dict = model(images, targets)
        else:
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)

        losses = torch.stack(list(loss_dict.values())).sum()

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            break

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()



def _val_one_epoch(model, val_data, device, history, metric: Metric):
    model.eval()
    pbar = tqdm(val_data)

    with torch.no_grad():
        for images, targets in pbar:
            images = list(image.to(device) for image in images)

            outputs = model(images)

            outputs = [{k: v.detach().cpu() for k, v in t.items()} for t in outputs]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(outputs, targets)

    outcomes = metric.compute()

    history.add_step_outcomes("val", **outcomes)



def train_model(model, batch_size, epochs, learning_rate, weight_decay, optimizer_class, device,
                metric = MeanAveragePrecision(), checkpoint_every: int = 5, target_metric = "map",
                target_res = (512, 512), num_workers = 0):

    optimizer = optimizer_class(model.parameters(), lr=learning_rate, weight_decay=weight_decay, )

    dataset_train = FireDatasetForDetection(train=True, target_resolution=target_res)
    dataset_val = FireDatasetForDetection(train=False, target_resolution=target_res)

    train_data = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers = num_workers)
    val_data = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers = num_workers)

    history = ModelHistory(["train", "val"], model.__class__.__name__, metric)
    prev_metric_val = None

    for epoch in range(epochs):
        _train_one_epoch(model, train_data, device, optimizer)
        _val_one_epoch(model, val_data, device, history, metric)

        if epoch % checkpoint_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, MODEL_DIR / f"model_{epoch}.pth")

        curr_metric_val = history.return_targeted_metric_val("val", target_metric)

        if prev_metric_val is None or \
        is_better_value(prev_metric_val, curr_metric_val, metric):

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            torch.save(checkpoint, MODEL_DIR / f"best_model.pth")

            prev_metric_val = curr_metric_val
            print(f"Found new best model at epoch {epoch} with metric {target_metric} and value {curr_metric_val}")

        metric.reset()

    return model, history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = fasterrcnn_resnet50_fpn(weights='DEFAULT', min_size = 256, max_size = 256)

    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    batch_size = 16
    epochs = 3
    learning_rate = 0.001
    weight_decay = 0.0005

    trained_model, history = train_model(
        model=model,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_class=torch.optim.SGD,
        device=device,
        target_metric="map_50",
        checkpoint_every=5,
        target_res=(512, 512),
        num_workers=4
    )

    print(history.history)

