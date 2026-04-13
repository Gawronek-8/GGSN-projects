from ultralytics import YOLO
from config import DATA_DIR

def epoch_status(trainer):
    current_epoch = trainer.epoch
    current_map50 = trainer.metrics.get('metrics/mAP50(B)', 0.0)
    if hasattr(trainer, 'tloss'):
        train_loss = trainer.tloss.sum().item()
    else:
        train_loss = 0.0

    print(f"\n Epoch: {current_epoch} | Loss: {train_loss:.4f} | mAP50: {current_map50:.4f}")

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.add_callback('on_fit_epoch_end', epoch_status)
    results = model.train(data="fire.yaml", epochs = 10, batch = 16, patience = 3, save=True, save_period = 2, imgsz = 512,
                project = str(DATA_DIR / "yolo"), name="test", workers = 4, cos_lr = True)

