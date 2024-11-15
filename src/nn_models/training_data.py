# import os
# import pandas as pd
# import numpy as np


# def load_trends_files(trends_dir):
#     """
#     Load and return all Google Trends files as a sorted list of DataFrames.
#     """
#     # Construct the full path from the current file's location
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     trends_dir = os.path.join(base_dir, trends_dir)

#     if not os.path.exists(trends_dir):
#         raise FileNotFoundError(f"The directory {trends_dir} does not exist.")
    
#     trend_files = sorted([os.path.join(trends_dir, f) 
#                           for f in os.listdir(trends_dir) if f.endswith('.csv')])
#     return [pd.read_csv(file, parse_dates=['date']).set_index('date') for file in trend_files]


# def load_stock_data(stock_file):
#     """
#     Load stock data from a CSV file into a pandas DataFrame.
#     """
#     # Set base directory as `src` for consistent path construction
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     stock_file_path = os.path.join(base_dir, stock_file)

#     if not os.path.exists(stock_file_path):
#         raise FileNotFoundError(f"The file {stock_file_path} does not exist.")

#     # Load stock data
#     stock_data = pd.read_csv(stock_file_path, parse_dates=['Date']).set_index('Date')
#     return stock_data[['Close', 'Volume']]


# def get_first_and_last_date(trends_df):
#     """
#     Get the first and last dates from the DataFrame index.

#     Parameters:
#     - trends_df (pd.DataFrame): The Google Trends DataFrame.

#     Returns:
#     - tuple: A tuple of the first and last dates as strings.
#     """
#     first_date = trends_df.index[0].strftime('%Y-%m-%d')
#     last_date = trends_df.index[-1].strftime('%Y-%m-%d')
#     return first_date, last_date


# def get_stock_window(stock_df, start_date, end_date):
#     """
#     Get the stock data within the specified date range.

#     Parameters:
#     - stock_df (pd.DataFrame): The stock data DataFrame.
#     - start_date (str): The start date of the window.
#     - end_date (str): The end date of the window.

#     Returns:
#     - pd.DataFrame: The stock data within the specified date range.
#     """
#     return stock_df[start_date:end_date]


# def create_training_data(stock_df, trends_df_list):
#     """
#     Create training data for a neural network model.
#     """
#     X, y = [], []  # Initialize lists to store input and output data

#     # Iterate over each Google Trends DataFrame except the last (as it has no subsequent file)
#     for i in range(len(trends_df_list) - 1):
#         trends_df = trends_df_list[i]
#         next_trends_df = trends_df_list[i + 1]

#         # Convert index to datetime for consistency
#         trends_df.index = pd.to_datetime(trends_df.index)

#         # Get the date range and corresponding stock data window
#         first_date, last_date = trends_df.index.min(), trends_df.index.max()
#         next_day = last_date + pd.Timedelta(days=1)
        
#         if next_day in stock_df.index and next_day in next_trends_df.index:
#             # Get stock window for 30-day input
#             stock_window = get_stock_window(stock_df, first_date, last_date)
            
#             # Concatenate stock and Google Trends data to create the input
#             input_data = pd.concat([stock_window, trends_df], axis=1)
#             X.append(input_data.values)

#             # Create target output for next day
#             stock_next_day_values = stock_df.loc[next_day][['Close', 'Volume']].values
#             google_trends_next_day_value = next_trends_df.loc[next_day].values
#             target_output = np.append(stock_next_day_values, google_trends_next_day_value)
#             y.append(target_output)

#     return np.array(X), np.array(y)



# trends_df_list = load_trends_files('data/google_trends/uber_stock')
# stock_df = load_stock_data(
#     'data/yahoo_finance/UBER_2021-12-31_to_2024-10-29.csv')


# x, y = create_training_data(stock_df, trends_df_list)

# print(x.shape)
# print(y.shape)

# def print_n_training_data_examples(x, y, n=2 ):
#     for i in range(n):
#         print(f"Example {i + 1}")
#         print("Input (X):")
#         print(x[i])
#         print("Output (y):")
#         print(y[i])
#         print()

# # print(x.shape)
# # print(y.shape)

# print_n_training_data_examples(x, y, n=3)

# import torch
# from torch.utils.data import DataLoader, TensorDataset

# # Convert numpy arrays to PyTorch tensors
# X_tensor = torch.tensor(x, dtype=torch.float32)  # shape (1002, 30, 3)
# y_tensor = torch.tensor(y, dtype=torch.float32)  # shape (1002, 3)

# # Create a dataset and DataLoader
# dataset = TensorDataset(X_tensor, y_tensor)
# batch_size = 32
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# import torch.nn as nn

# class StockPredictionLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(StockPredictionLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)  # Linear layer for output

#     def forward(self, x):
#         # Pass through LSTM layers
#         out, _ = self.lstm(x)  # `out` shape: (batch_size, sequence_length, hidden_size)
#         out = self.fc(out[:, -1, :])  # Use the last time step's output
#         return out
    

# input_size = 3
# hidden_size = 64
# num_layers = 2
# output_size = 3

# model = StockPredictionLSTM(input_size, hidden_size, num_layers, output_size)

# # Loss function and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
# # Training loop
# epochs = 100
# for epoch in range(epochs):
#     model.train()
#     train_loss = 0

#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()           # Clear gradients from the previous step
#         output = model(X_batch)         # Forward pass
#         loss = criterion(output, y_batch)  # Calculate loss
#         loss.backward()                 # Backpropagation
#         optimizer.step()                # Update weights
#         train_loss += loss.item() * X_batch.size(0)  # Accumulate loss

#     train_loss /= len(train_loader.dataset)  # Average loss for the epoch

#     # Print progress
#     if epoch % 10 == 0:
#         print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")
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

    stock_data = pd.read_csv(stock_file_path, parse_dates=['Date']).set_index('Date')
    return stock_data[['Close', 'Volume']]

def get_stock_window(stock_df, start_date, end_date):
    return stock_df[start_date:end_date]

# def create_training_data(stock_df, trends_df_list):
#     X, y = [], []  # Initialize lists to store input and output data
#     for i in range(len(trends_df_list) - 1):
#         trends_df = trends_df_list[i]
#         next_trends_df = trends_df_list[i + 1]

#         # Get the date range and corresponding stock data window
#         first_date, last_date = trends_df.index.min(), trends_df.index.max()
#         next_day = last_date + pd.Timedelta(days=1)
        
#         if next_day in stock_df.index and next_day in next_trends_df.index:
#             stock_window = get_stock_window(stock_df, first_date, last_date)
#             input_data = pd.concat([stock_window, trends_df], axis=1)
#             X.append(input_data.values)

#             # Create target output for next day
#             stock_next_day_values = stock_df.loc[next_day][['Close', 'Volume']].values
#             google_trends_next_day_value = next_trends_df.loc[next_day].values
#             target_output = np.append(stock_next_day_values, google_trends_next_day_value)
#             y.append(target_output)

#     return np.array(X), np.array(y)

def create_training_data(stock_df, trends_df_list):
    X, y = [], []

    # Set up scalers for stock and Google Trends data
    stock_scaler = MinMaxScaler()
    trends_scaler = MinMaxScaler()

    # Fit the scalers to transform both stock data and trends data
    stock_df[['Close', 'Volume']] = stock_scaler.fit_transform(stock_df[['Close', 'Volume']])
    for i in range(len(trends_df_list)):
        trends_df_list[i] = trends_df_list[i].apply(lambda col: trends_scaler.fit_transform(col.values.reshape(-1, 1)).flatten())
    
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
            target_output = np.append(stock_next_day_values, google_trends_next_day_value)
            y.append(target_output)

    return np.array(X), np.array(y)


# Load data and prepare tensors
trends_df_list = load_trends_files('data/google_trends/uber_stock')
stock_df = load_stock_data('data/yahoo_finance/UBER_2021-12-31_to_2024-10-29.csv')
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer for output

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step's output
        return out

# Model parameters
input_size = 3
hidden_size = 128
num_layers = 4
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
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

import matplotlib.pyplot as plt

# Set the model to evaluation mode
model.eval()

# Generate rolling 1-day predictions on the test set
one_day_ahead_preds = []
# actual_closing_prices = y_test[:, 0].numpy()  # Actual closing prices in the test set

# with torch.no_grad():
#     for i in range(len(X_test)):
#         # Use the actual 30-day window in the test set to predict the next day
#         current_window = X_test[i].unsqueeze(0)  # Shape: (1, 30, 3)
#         prediction = model(current_window).numpy()[0]  # Get prediction as numpy array

#         # Store the closing price prediction
#         one_day_ahead_preds.append(prediction[0])  # Only store the closing price for 1-day ahead

# # Convert predictions to numpy array for easier plotting
# one_day_ahead_preds = np.array(one_day_ahead_preds)

# # Get test dates
# test_dates = stock_df.index[-len(y_test):]

# # Plot actual closing prices and rolling 1-day-ahead predictions
# plt.figure(figsize=(12, 6))

# # Plot actual closing prices for the entire test set
# plt.plot(test_dates, actual_closing_prices, label='Actual Closing Price', color='blue')

# # Plot rolling 1-day-ahead predictions
# plt.plot(test_dates, one_day_ahead_preds, label='1-Day Ahead Predicted Closing Price', color='green', linestyle='dashed')

# # Add labels and title
# plt.xlabel('Date')
# plt.ylabel('Closing Price')
# plt.title('Rolling 1-Day-Ahead Closing Price Prediction')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()
# Convert predictions to numpy array for easier plotting
# Ensure one_day_ahead_preds matches the length of the test set
if len(one_day_ahead_preds) == 0:
    print("Error: No predictions were generated. Check the prediction loop for issues.")
else:
    # Convert predictions to numpy array for easier plotting
    one_day_ahead_preds = np.array(one_day_ahead_preds)

    # Concatenate actual closing prices for the entire dataset (training + test)
    all_closing_prices = np.concatenate([y_train.numpy()[:, 0], y_test.numpy()[:, 0]])
    
    # Get dates for the entire dataset and for the test set specifically
    all_dates = stock_df.index[-len(all_closing_prices):]
    test_dates = stock_df.index[-len(y_test):]

    # Plot actual closing prices and rolling 1-day-ahead predictions for the test set
    plt.figure(figsize=(12, 6))

    # Plot actual closing prices for the entire dataset (training + test)
    plt.plot(all_dates, all_closing_prices, label='Actual Closing Price', color='blue')

    # Plot 1-day-ahead predictions for the test set
    plt.plot(test_dates, one_day_ahead_preds, label='1-Day Ahead Predicted Closing Price (Test Set)', color='green', linestyle='dashed')

    # Add labels, title, and legend
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.title('Actual Closing Price with 1-Day-Ahead Predictions on Test Set')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


