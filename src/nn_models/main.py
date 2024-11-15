from data_loader import create_dataloader
from lstm_model import StockTrendsLSTM
from train import train_model
from utils import plot_recursive_predictions

def get_test_data(test_loader):
    """
    Extracts the initial sequence and actual values from the test DataLoader.

    Parameters:
    - test_loader: DataLoader containing test data.

    Returns:
    - tuple: (initial_sequence, actual_data), where initial_sequence is the first
             sequence in the test set, and actual_data contains the real values for comparison.
    """
    for X, y in test_loader:
        initial_sequence = X[0].unsqueeze(0)  # Shape (1, 30, 3)
        actual_data = y.numpy()  # Convert all y data in test_loader to numpy array
        return initial_sequence, actual_data

def main(stock_file, trends_dir, epochs=10, batch_size=32, hidden_size=128, num_layers=2):
    # Create dataloaders for training and testing
    train_loader, test_loader = create_dataloader(stock_file, trends_dir, batch_size=batch_size)

    # Initialize and train the model
    model = StockTrendsLSTM(hidden_size=hidden_size, num_layers=num_layers)
    print("Starting training...")
    train_model(model, train_loader, epochs=epochs)

    # Perform recursive predictions and plot
    test_sequence, actual_data = get_test_data(test_loader)
    plot_recursive_predictions(model, test_sequence, actual_data, days=30)

if __name__ == "__main__":
    stock_file = 'data/yahoo_finance/UBER_2021-12-31_to_2024-10-29.csv'
    trends_dir = 'data/google_trends/uber_stock'
    main(stock_file, trends_dir)
