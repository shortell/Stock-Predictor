import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
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
    return stock_data[['Close', 'Volume']]

def get_stock_window(stock_df, start_date, end_date):
    return stock_df[start_date:end_date]

def create_training_data(stock_df, trends_df_list):
    X, y = [], []
    stock_scaler = MinMaxScaler()
    trends_scaler = MinMaxScaler()

    # Scale stock data and trends data
    stock_df[['Close', 'Volume']] = stock_scaler.fit_transform(stock_df[['Close', 'Volume']])
    trends_df_list = [df.apply(lambda col: trends_scaler.fit_transform(col.values.reshape(-1, 1)).flatten()) for df in trends_df_list]
    
    for i in range(len(trends_df_list) - 1):
        trends_df = trends_df_list[i]
        next_trends_df = trends_df_list[i + 1]
        first_date, last_date = trends_df.index.min(), trends_df.index.max()
        next_day = last_date + pd.Timedelta(days=1)
        
        if next_day in stock_df.index and next_day in next_trends_df.index:
            stock_window = get_stock_window(stock_df, first_date, last_date)
            input_data = pd.concat([stock_window, trends_df], axis=1)
            X.append(input_data.values)

            stock_next_day_values = stock_df.loc[next_day][['Close', 'Volume']].values
            google_trends_next_day_value = next_trends_df.loc[next_day].values
            y.append(np.append(stock_next_day_values, google_trends_next_day_value))

    return np.array(X), np.array(y)

# Load and prepare data
trends_df_list = load_trends_files('data/google_trends/uber_stock')
stock_df = load_stock_data('data/yahoo_finance/UBER_2021-12-31_to_2024-10-29.csv')
X, y = create_training_data(stock_df, trends_df_list)

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

# Define LSTM Model
class StockPredictionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPredictionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer for output

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out

# Initialize model, loss function, and optimizer
model = StockPredictionLSTM(input_size=3, hidden_size=128, num_layers=4, output_size=3)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Validation
epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            val_loss += criterion(output, y_batch).item() * X_batch.size(0)
    val_loss /= len(test_loader.dataset)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Generate rolling 1-day predictions on the test set
model.eval()
one_day_ahead_preds = []
with torch.no_grad():
    for i in range(len(X_test)):
        current_window = X_test[i].unsqueeze(0)  # Shape: (1, 30, 3)
        prediction = model(current_window).numpy()[0]
        one_day_ahead_preds.append(prediction[0])  # Only store the closing price

# Plot the results
one_day_ahead_preds = np.array(one_day_ahead_preds)
all_closing_prices = np.concatenate([y_train.numpy()[:, 0], y_test.numpy()[:, 0]])
all_dates = stock_df.index[-len(all_closing_prices):]
test_dates = stock_df.index[-len(y_test):]

plt.figure(figsize=(12, 6))
plt.plot(all_dates, all_closing_prices, label='Actual Closing Price', color='blue')
plt.plot(test_dates, one_day_ahead_preds, label='1-Day Ahead Predicted Closing Price (Test Set)', color='red')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual Closing Price with 1-Day-Ahead Predictions on Test Set')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
