import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import REGION_SLOP

criterion = nn.MSELoss()
LOG_INTERVAL = 100

def train_model(model: nn.Module, dataloader: DataLoader, learning_rate: float, device: str, saved_model_file: str, epochs:int = 1000):
    start = time.time()
    size = len(dataloader.dataset)  # type: ignore
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        print(f"epoch {epoch} / {epochs}")
        for batch_id, batch in enumerate(dataloader):
            batch_features = batch[0].to(device)
            batch_labels = batch[1].to(device)
            output = model(batch_features)

            center_outputs = output[:, REGION_SLOP: -1*REGION_SLOP]
            center_labels = batch_labels[:, REGION_SLOP: -1*REGION_SLOP]

            mask = (center_outputs != 0) | (center_labels != 0)
            loss = criterion(center_outputs[mask], center_labels[mask])

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            if batch_id % LOG_INTERVAL == 0:
                loss = total_loss / LOG_INTERVAL 
                total_loss = 0
                current = batch_id + 1
                print(f"loss: {loss:>10f}  Batch [{current:>5d}/{size:>5d}]")
                print(f"{int((time.time() - start) / 60)} minutes have elapsed")
                print("Saving model")
                torch.save(model.state_dict(), f"{saved_model_file}")
        