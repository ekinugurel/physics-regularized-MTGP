import torch.nn as nn
import torch
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Apply dropout
        last_out = self.dropout(lstm_out)

        # Fully connected layer on the last time step's output
        out = self.fc(last_out)
        
        return out
    
def LSTM_train(num_epochs, data_loader, model, loss_function, optimizer, verbose=True):
    ls = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in data_loader:
            # Reset gradients
            optimizer.zero_grad()
        
            # Forward pass
            outputs = model(inputs)
        
            # Compute loss
            loss = loss_function(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        
            # Accumulate loss
            epoch_loss += loss.item()

        # Save loss for the epoch
        ls.append(epoch_loss / len(data_loader))

        if verbose:
            # Print average loss for the epoch
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(data_loader)}')

    return model, ls

def MC_LSTM(model, x, num_runs=30):
    # Set model to training mode to enable dropout during inference
    model.train()

    # Collect predictions
    predictions = []

    # Perform Monte Carlo simulations
    with torch.no_grad():
        for _ in range(num_runs):
            predictions.append(model(x).numpy())

    # Convert predictions to a numpy array
    predictions = np.array(predictions)

    # Calculate mean and standard deviation
    mean_pred = np.mean(predictions, axis=0)
    std_dev_pred = np.std(predictions, axis=0)

    # Calculate 95% confidence interval
    z_score = 1.96  # Z-score for 95% confidence interval
    lower_bound = mean_pred - z_score * std_dev_pred
    upper_bound = mean_pred + z_score * std_dev_pred

    return mean_pred, lower_bound, upper_bound, std_dev_pred
