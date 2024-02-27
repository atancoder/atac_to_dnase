from tabnanny import check
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import REGION_SLOP
from typing import List

criterion = nn.MSELoss()
LOG_INTERVAL = 100
EPOCH_STOP_THRESHOLD = 50

def train_model(model: nn.Module, dataloader: DataLoader, learning_rate: float, device: str, saved_model_file: str, epochs: int, checkpoint_model: bool) -> List[float]:
    start_time = time.time()
    size = len(dataloader.dataset)  # type: ignore
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    

    best_loss = float('inf')
    epochs_since_best_loss = 0
    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        interval_loss = 0
        print(f"Epoch {epoch} / {epochs}\n")
        for idx, batch in enumerate(dataloader):
            batch_id = idx + 1
            batch_features = batch[0].to(device)
            batch_labels = batch[1].to(device)
            output = model(batch_features)

            center_outputs = output[:, REGION_SLOP: -1*REGION_SLOP]
            center_labels = batch_labels[:, REGION_SLOP: -1*REGION_SLOP]
            loss = criterion(center_outputs.sum(dim=1), center_labels.sum(dim=1))

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            interval_loss += loss.item()
            if batch_id % LOG_INTERVAL == 0:
                current = batch_id * batch_features.shape[0]
                print(f"Processed [{current:>5d}/{size:>5d}] samples")
                epoch_loss += interval_loss
                avg_interval_loss = interval_loss / LOG_INTERVAL
                print(f"Avg loss over {LOG_INTERVAL} batches: {avg_interval_loss:>10f}")
                interval_loss = 0

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_since_best_loss = 0
            if checkpoint_model:
                torch.save(model.state_dict(), f"{saved_model_file}")
                print("Checkpointed model")
        else:
            epochs_since_best_loss += 1
            if epochs_since_best_loss == EPOCH_STOP_THRESHOLD:
                print(f"Haven't beat the loss in {EPOCH_STOP_THRESHOLD} epochs. Stopping early")
                return epoch_losses
            epoch_losses.append(epoch_loss)

        print(f"Epoch complete. {int((time.time() - start_time) / 60)} minutes have elapsed")
    return epoch_losses
        