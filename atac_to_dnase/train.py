from tabnanny import check
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import REGION_SLOP

criterion = nn.MSELoss()
LOG_INTERVAL = 100

def train_model(model: nn.Module, dataloader: DataLoader, learning_rate: float, device: str, saved_model_file: str, epochs: int, checkpoint_model: bool):
    start = time.time()
    size = len(dataloader.dataset)  # type: ignore
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch} / {epochs}\n")
        for batch_id, batch in enumerate(dataloader):
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

            total_loss += loss.item()
            if batch_id % LOG_INTERVAL == 0:
                loss = total_loss / LOG_INTERVAL 
                total_loss = 0
                current = batch_id + 1
                print(f"Avg loss over {LOG_INTERVAL} batches: {loss:>10f}")
                print(f"Processed [{current:>5d}/{size:>5d}] samples")
                print(f"{int((time.time() - start) / 60)} minutes have elapsed")
                if checkpoint_model:
                    print("Checkpointing model")
                    torch.save(model.state_dict(), f"{saved_model_file}")
        