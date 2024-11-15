import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the LSTM model
class StockTrendsLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2, output_size=3):
        """
        LSTM model for predicting the next day's stock close, volume, and trends.
        
        Parameters:
        - input_size: Number of features per time step (3: Close, Volume, Trends).
        - hidden_size: Number of neurons in each LSTM layer.
        - num_layers: Number of LSTM layers.
        - output_size: Number of output values (3: Close, Volume, Trends).
        """
        super(StockTrendsLSTM, self).__init__()
        
        # LSTM layer(s)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, sequence_length, hidden_size)
        
        # Only take the output from the last time step
        out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through fully connected layer
        out = self.fc(out)  # Shape: (batch_size, output_size)
        
        return out

# Initialize the model
model = StockTrendsLSTM()

# Print model structure
print(model)
