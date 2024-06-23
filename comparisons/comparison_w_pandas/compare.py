import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


# Function to read and preprocess the CSV files
def read_results(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces in column names
        model_name = os.path.basename(file).split('.')[0]
        df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# Function to plot comparison graphs and save them
def plot_comparison(df, metric, save_dir):
    pivot_df = df.pivot(index='epoch', columns='model', values=metric)
    pivot_df.dropna(how='all', inplace=True)  # Drop rows where all elements are NaN

    plt.figure(figsize=(12, 8))
    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], label=column, linewidth=2, marker='o')
        plt.fill_between(pivot_df.index, pivot_df[column], alpha=0.1)

    plt.title(f'Comparison of {metric} over epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.grid(True)
    plt.legend(title='Model', fontsize=12, loc='lower right')

    # Save the plot
    plot_path = os.path.join(save_dir, f'{metric.replace("/", "_")}.png')
    plt.savefig(plot_path)
    plt.close()


# List all CSV files in the current directory
files = glob.glob("*.csv")

# Read and preprocess the CSV files
df = read_results(files)

# Print the columns of the dataframe to inspect the structure
print("Columns in the dataframe:", df.columns)
print(df.head())

# Define the metrics to be compared
metrics = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
           'val/box_loss', 'val/cls_loss']

# Create the directory to save plots
save_dir = './model_plots_ongoing1'
os.makedirs(save_dir, exist_ok=True)

# Plot comparison graphs for each metric
for metric in metrics:
    if metric in df.columns:
        plot_comparison(df, metric, save_dir)
    else:
        print(f"Metric {metric} not found in the dataframe.")
