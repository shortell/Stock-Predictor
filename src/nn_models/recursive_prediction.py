import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Functions to load and prepare data (no changes needed)


def load_trends_files(trends_dir):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trends_dir = os.path.join(base_dir, trends_dir)

    if not os.path.exists(trends_dir):
        raise FileNotFoundError(f"The directory {trends_dir} does not exist.")

    trend_files = sorted([os.path.join(trends_dir, f)
                          for f in os.listdir(trends_dir) if f.endswith('.csv')])
    return [pd.read_csv(file, parse_dates=['date']).set_index('date') for file in trend_files]


def load_stock_data(stock_file):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stock_file_path = os.path.join(base_dir, stock_file)

    if not os.path.exists(stock_file_path):
        raise FileNotFoundError(f"The file {stock_file_path} does not exist.")

    stock_data = pd.read_csv(stock_file_path, parse_dates=[
                             'Date']).set_index('Date')
    return stock_data[['Close', 'Volume']]


def get_stock_window(stock_df, start_date, end_date):
    return stock_df[start_date:end_date]


def create_training_data(stock_df, trends_df_list):
    X, y = [], []

    # Set up scalers for stock and Google Trends data
    stock_scaler = MinMaxScaler()
    trends_scaler = MinMaxScaler()

    # Fit the scalers to transform both stock data and trends data
    stock_df[['Close', 'Volume']] = stock_scaler.fit_transform(
        stock_df[['Close', 'Volume']])
    for i in range(len(trends_df_list)):
        trends_df_list[i] = trends_df_list[i].apply(
            lambda col: trends_scaler.fit_transform(col.values.reshape(-1, 1)).flatten())

    for i in range(len(trends_df_list) - 1):
        trends_df = trends_df_list[i]
        next_trends_df = trends_df_list[i + 1]

        first_date, last_date = trends_df.index.min(), trends_df.index.max()
        next_day = last_date + pd.Timedelta(days=1)

        if next_day in stock_df.index and next_day in next_trends_df.index:
            stock_window = get_stock_window(stock_df, first_date, last_date)
            input_data = pd.concat([stock_window, trends_df], axis=1)
            X.append(input_data.values)

            stock_next_day_values = stock_df.loc[next_day][[
                'Close', 'Volume']].values
            google_trends_next_day_value = next_trends_df.loc[next_day].values
            target_output = np.append(
                stock_next_day_values, google_trends_next_day_value)
            y.append(target_output)

    return np.array(X), np.array(y)


# Load data and prepare tensors
trends_df_list = load_trends_files('data/google_trends/uber_stock')
stock_df = load_stock_data(
    'data/yahoo_finance/UBER_2021-12-31_to_2024-10-29.csv')
x, y = create_training_data(stock_df, trends_df_list)

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# === Step 1: Perform 80-20 Split ===
split_idx = int(0.8 * len(X_tensor))
X_train, X_test = X_tensor[:split_idx], X_tensor[split_idx:]
y_train, y_test = y_tensor[:split_idx], y_tensor[split_idx:]

# === Step 2: Create DataLoaders for Train and Test Sets ===
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the LSTM Model


class StockPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        # Linear layer for output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out


# Model parameters
input_size = 3
hidden_size = 64
num_layers = 2
output_size = 3

model = StockPredictionLSTM(input_size, hidden_size, num_layers, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Step 3: Training Loop with Validation ===
epochs = 100
for epoch in range(epochs):
    model.train()  # Set model to training mode
    train_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item() * X_batch.size(0)

    val_loss /= len(test_loader.dataset)

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch {
              epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


# Set the model to evaluation mode
# Use last 30-day window from training data to initiate recursive predictions
initial_window = X_test[0].unsqueeze(0)  # Shape: (1, 30, 3)
model.eval()

# Storage for recursive predictions
recursive_preds = []

with torch.no_grad():
    current_window = initial_window
    for _ in range(len(y_test)):
        # Predict the next day’s closing price, volume, and search volume
        prediction = model(current_window).numpy()[
            0]  # Get prediction as numpy array

        # Append only the closing price to recursive predictions
        recursive_preds.append(prediction[0])  # Only store the closing price

        # Prepare the new input by shifting the window
        next_input = np.vstack(
            [current_window.numpy()[0, 1:, :], prediction.reshape(1, -1)])
        current_window = torch.tensor(
            next_input, dtype=torch.float32).unsqueeze(0)


# Combine training and test closing prices for the full dataset
all_closing_prices = np.concatenate(
    [y_train.numpy()[:, 0], y_test.numpy()[:, 0]])
# Use dates for the last len(all_closing_prices) rows
dates = stock_df.index[-len(all_closing_prices):]

# Plot the full dataset's actual closing prices
plt.figure(figsize=(12, 6))
plt.plot(dates, all_closing_prices, label='Actual Closing Price', color='blue')

# Plot predictions on the last 20%
test_dates = stock_df.index[-len(y_test):]
plt.plot(test_dates, recursive_preds,
         label='Predicted Closing Price (Last 20%)', color='red')

# Formatting and labels
plt.title('Closing Price Prediction for Entire Dataset')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
