import torch
from tqdm import tqdm
import torch.optim as optim
from utils import NegativeLogLikelihood
import numpy as np

# define your loss
criterion = NegativeLogLikelihood()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 1000
best_val_loss = float('inf')
save_path = 'your_path'

for epoch in range(num_epochs):
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch [{epoch+1}/{num_epochs}] - Learning Rate: {current_lr:.8f}")

    model.train()
    epoch_loss = 0.0
    batch_count = 0

    # Iterate over each group loader (each corresponding to a specific agent count)
    for loader in tqdm(train_loaders, desc=f"Epoch {epoch+1}/{num_epochs} - Training Groups", leave=False):
        for batch_source, batch_target in tqdm(loader, desc="Training Batches", leave=False):
            
            batch_target = batch_target.to(device) 
            optimizer.zero_grad()

            mu, sigma, _,_ = model(batch_source)  
            loss = criterion(mu, sigma, batch_target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

    avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')

    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.8f}")
    
