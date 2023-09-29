import pandas as pd
import numpy as np

# Create an empty DataFrame to store the time series data
df = pd.DataFrame(columns=['timestamp', 'value', 'labels'])

# Define the number of windows to generate
num_windows = 100

# Define your threshold and window size
threshold = 7

# Function to label windows
def label_windows(df, threshold):
    labels = []
    for i in range(len(df)):
        if df['value'].iloc[i] >= threshold:
            labels.append(1)  # Add a single label for each event
        else:
            labels.append(0)
    return labels

# Generate time series windows with varying sizes and periods
for _ in range(num_windows):
    window_size = np.random.randint(5, 15)  # Random window size between 5 and 15
    window_period = np.random.choice(['H', 'D', 'W'])  # Random period (hourly, daily, weekly)

    # Generate time series data for the current window
    data = {
        'timestamp': pd.date_range(start='2023-09-01', periods=window_size, freq=window_period),
        'value': np.random.randint(0, 10, size=window_size)
    }
    window_df = pd.DataFrame(data)

    # Label the current window
    window_df['labels'] = label_windows(window_df, threshold)

    # Append the labeled window to the main DataFrame
    df = df.append(window_df, ignore_index=True)

# Add a buffer of 30 data points before and after each event window
buffered_windowed_data = []

for i in range(len(df)):
    if df['labels'].iloc[i] == 1:
        start_idx = max(0, i - 30)
        end_idx = min(len(df), i + 30)
        buffered_windowed_data.append(df.iloc[start_idx:end_idx])

# Print the first few windows with the buffer
for i, window in enumerate(buffered_windowed_data[:5]):
    print(f"Window {i} (Label: {window.iloc[0]['labels']}):")
    print(window.drop(columns='labels'))
    print()
