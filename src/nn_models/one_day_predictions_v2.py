import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Functions to load and prepare data
def load_trends_files(trends_dir):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trends_dir = os.path.join(base_dir, trends_dir)
    if not os.path.exists(trends_dir):
        raise FileNotFoundError(f"The directory {trends_dir} does not exist.")
    trend_files = sorted([os.path.join(trends_dir, f) for f in os.listdir(trends_dir) if f.endswith('.csv')])
    return [pd.read_csv(file, parse_dates=['date']).set_index('date') for file in trend_files]

def load_stock_data(stock_file):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stock_file_path = os.path.join(base_dir, stock_file)
    if not os.path.exists(stock_file_path):
        raise FileNotFoundError(f"The file {stock_file_path} does not exist.")
    stock_data = pd.read_csv(stock_file_path, parse_dates=['Date']).set_index('Date')
    return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

def get_stock_window(stock_df, start_date, end_date):
    return stock_df[start_date:end_date]

def create_training_data(stock_df, trends_df_list):
    X, y = [], []

    # Set up scalers for stock and Google Trends data
    stock_scaler = StandardScaler()
    trends_scaler = StandardScaler()

    # Scale stock data and apply StandardScaler
    stock_df[['Open', 'High', 'Low', 'Close', 'Volume']] = stock_scaler.fit_transform(stock_df[['Open', 'High', 'Low', 'Close', 'Volume']])
    trends_df_list = [df.apply(lambda col: trends_scaler.fit_transform(col.values.reshape(-1, 1)).flatten()) for df in trends_df_list]

    # Construct training data with sliding window approach
    for i in range(len(trends_df_list) - 1):
        trends_df = trends_df_list[i]
        next_trends_df = trends_df_list[i + 1]

        first_date, last_date = trends_df.index.min(), trends_df.index.max()
        next_day = last_date + pd.Timedelta(days=1)
        
        if next_day in stock_df.index and next_day in next_trends_df.index:
            stock_window = get_stock_window(stock_df, first_date, last_date)
            input_data = pd.concat([stock_window, trends_df], axis=1)
            X.append(input_data.values)

            # Target output for the next day with all 7 variables
            stock_next_day_values = stock_df.loc[next_day][['Open', 'High', 'Low', 'Close', 'Volume']].values
            google_trends_next_day_values = next_trends_df.loc[next_day].values[:2]  # Two search terms
            target_output = np.concatenate([stock_next_day_values, google_trends_next_day_values])
            y.append(target_output)

    return np.array(X), np.array(y)

# Load data and prepare tensors
trends_df_list = load_trends_files('data/google_trends/best_buy_stock_bby')  # Two search term files
stock_df = load_stock_data('data/yahoo_finance/BBY_2021-01-01_to_2024-10-29.csv')
X, y = create_training_data(stock_df, trends_df_list)

# Check for NaNs
if np.isnan(X).any() or np.isnan(y).any():
    raise ValueError("NaN values found in data. Check data processing steps.")

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split into training and test sets
split_idx = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Model parameters
input_size = 7  # 5 stock features + 2 Google Trends search terms
hidden_size = 128  # Increase hidden size
num_layers = 4  # Increase to 4 layers
dropout_rate = 0.3  # Increase dropout rate
output_size = 7  # Predicting all 7 variables


# class StockPredictionLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(StockPredictionLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(0.2)  # 20% dropout
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.dropout(out)  # Apply dropout after LSTM
#         out = self.fc(out[:, -1, :])
#         return out
class StockPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.3):
        super(StockPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)  # Apply dropout
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out)  # Apply dropout after LSTM
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out
    
model = StockPredictionLSTM(input_size, hidden_size, num_layers, output_size, dropout_rate)

# Define optimizer with weight decay for regularization
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Lower learning rate with weight decay
criterion = nn.MSELoss()  # Mean Squared Error for regression

# Training loop with validation and gradient clipping
epochs = 100
for epoch in range(epochs):
    model.train()  # Set the model to training mode
    train_loss = 0

    # Training loop
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Zero the gradients
        output = model(X_batch)  # Forward pass
        loss = criterion(output, y_batch)  # Calculate loss
        loss.backward()  # Backward pass
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients
        optimizer.step()  # Update weights
        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(test_loader.dataset)

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")



# Define the prediction horizon as half of the window size
window_size = X_test.shape[1]
# Define the prediction horizon length
# Define the prediction horizon as the length of the test set for continuous 1-day predictions
horizon = len(X_test)

# Perform continuous 1-day predictions without recursion
one_day_preds = []

# Make predictions on each day in the test set
model.eval()
with torch.no_grad():
    for i in range(horizon):
        # Take each test sample one by one
        input_data = X_test[i].unsqueeze(0)  # Shape (1, window_size, input_size)
        
        # Predict the next day's 7 variables
        prediction = model(input_data).numpy()[0]  # Get prediction as numpy array
        one_day_preds.append(prediction)  # Store all 7 predicted variables

# Extract dates and data for the first month (30 days) of the test period
month_length = 30  # Set the length of the period to display
actual_data = y_test.numpy()[:, 3][:month_length]  # Use only the 'Close' price for the first 30 days
test_dates = stock_df.index[-len(y_test):][:month_length]  # First 30 dates in the test period
predicted_closing_prices = [p[3] for p in one_day_preds[:month_length]]  # First 30 predictions

# Plot the actual closing prices for the first month of predictions
plt.figure(figsize=(12, 6))
plt.plot(test_dates, actual_data, label='Actual Closing Price', color='blue')

# Plot the one-day predictions for the first month as a line
plt.plot(test_dates, predicted_closing_prices, label='Predicted Closing Price (1-day predictions)', color='red')

# Formatting and labels
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Closing Price Prediction (First Month of 1-Day Predictions)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
