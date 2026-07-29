import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):
    """
    Train a PyTorch model and return training history.
    """

    # Step 1: Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Step 2: Create loss function
    criterion = nn.CrossEntropyLoss()

    history = []
    n_samples = X_train.shape[0]

    # Step 3: Training loop
    for epoch in range(epochs):
        model.train()

        # Shuffle training data
        indices = torch.randperm(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        total_loss = 0.0

        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            x_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(x_batch)

            # Compute loss
            loss = criterion(logits, y_batch)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()

            total_loss += loss.item() * len(x_batch)

        train_loss = total_loss / n_samples

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()

            predictions = val_logits.argmax(dim=1)
            val_accuracy = (predictions == y_val).float().mean().item()

        # Save history
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    return history