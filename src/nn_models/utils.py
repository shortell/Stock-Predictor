
import torch

import matplotlib.pyplot as plt

def test_model(model, sample_input):
    model.eval()
    with torch.no_grad():
        sample_input = sample_input.unsqueeze(0)
        prediction = model(sample_input)
    return prediction

def recursive_predict(model, initial_sequence, days=30):
    """
    Make predictions recursively up to a specified number of days.

    Parameters:
    - model: Trained LSTM model.
    - initial_sequence (torch.Tensor): The initial input sequence of shape (1, 30, 3).
    - days (int): Number of future days to predict.

    Returns:
    - list: Predicted values for the next `days` days.
    """
    model.eval()
    predictions = []
    current_sequence = initial_sequence.clone().unsqueeze(0)  # Ensure shape is (1, 30, 3)

    with torch.no_grad():
        for _ in range(days):
            next_day_prediction = model(current_sequence)  # Shape: (1, 3)
            predictions.append(next_day_prediction.squeeze(0).numpy())

            # Update sequence: remove the oldest day and add the new prediction
            next_input = torch.cat((current_sequence[:, 1:], next_day_prediction.unsqueeze(1)), dim=1)
            current_sequence = next_input

    return predictions

def plot_recursive_predictions(model, test_data, actual_data, days=30):
    """
    Plot recursive predictions against actual values.

    Parameters:
    - model: Trained LSTM model.
    - test_data (torch.Tensor): Initial 30-day input sequence for prediction.
    - actual_data (pd.DataFrame): DataFrame of actual values to compare against.
    - days (int): Number of days to predict into the future.
    """
    # Generate predictions
    predictions = recursive_predict(model, test_data, days=days)
    predicted_values = [pred[0] for pred in predictions]  # Extract close price predictions

    # Prepare the actual values for the same time period
    actual_close_prices = actual_data['Close'].values[:days]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(days), actual_close_prices, label="Actual Close Price", color="blue")
    plt.plot(range(days), predicted_values, label="Predicted Close Price", color="red", linestyle="--")
    plt.xlabel("Days")
    plt.ylabel("Close Price")
    plt.title("Recursive Predictions vs Actual Close Price")
    plt.legend()
    plt.show()
