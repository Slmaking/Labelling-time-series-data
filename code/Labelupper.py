import pandas as pd
import numpy as np

# Create an empty DataFrame to store the time series data
df = pd.DataFrame(columns=['timestamp', 'speed', 'labels', 'acc_x', 'acc_y', 'acc_z'])

# Define the number of windows to generate
num_windows = 100

# Define your speed deviation thresholds
lower_speed_deviation_threshold = -5
speed_deviation_threshold = 3

# Function to label windows based on speed deviation
def label_windows(df, threshold_low, threshold_high):
    labels = []
    for i in range(len(df)):
        speed_deviation = df['speed_deviation'].iloc[i]
        if speed_deviation >= threshold_high:
            labels.append(1)  # Add a label 1 for high deviation
        elif speed_deviation <= threshold_low:
            labels.append(2)  # Add a label 2 for low deviation
        else:
            labels.append(0)  # Add a label 0 for no event
    return labels

# Generate time series windows with varying sizes and periods
for _ in range(num_windows):
    window_size = np.random.randint(5, 15)  # Random window size between 5 and 15
    window_period = np.random.choice(['H', 'D', 'W'])  # Random period (hourly, daily, weekly)

    # Generate time series data for the current window
    data = {
        'timestamp': pd.date_range(start='2023-09-01', periods=window_size, freq=window_period),
        'speed': np.random.randint(0, 10, size=window_size)
    }
    window_df = pd.DataFrame(data)

    # Calculate speed deviation for the current window
    window_df['speed_deviation'] = window_df['speed'].diff().abs().mean()

    # Label the current window based on speed deviation
    window_df['labels'] = label_windows(window_df, lower_speed_deviation_threshold, speed_deviation_threshold)

    # Generate random values for acc_x, acc_y, and acc_z
    window_df['acc_x'] = np.random.rand(window_size)
    window_df['acc_y'] = np.random.rand(window_size)
    window_df['acc_z'] = np.random.rand(window_size)

    # Append the labeled window to the main DataFrame
    df = df.append(window_df, ignore_index=True)

# Add a buffer of 3 data points before and after each event window
buffered_windowed_data = []

for i in range(len(df)):
    if df['labels'].iloc[i] == 1 or df['labels'].iloc[i] == 2:
        start_idx = max(0, i - 30)
        end_idx = min(len(df), i + 30)
        buffered_windowed_data.append(df.iloc[start_idx:end_idx])

# Print the first few windows with the buffer
for i, window in enumerate(buffered_windowed_data[:5]):
    print(f"Window {i} (Label: {window.iloc[0]['labels']}):")
    print(window.drop(columns='labels'))
    print()
