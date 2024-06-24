import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.animation as animation

"""
Function to read and preprocess the CSV files.
"""


def read_results(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces in column names
        model_name = os.path.basename(file).split('.')[0]
        df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


"""
Function to create an animation for the mAP50-95(B) metric.
"""


def animate_comparison(df, metric, save_dir, duration=7):
    pivot_df = df.pivot(index='epoch', columns='model', values=metric)
    pivot_df.dropna(how='all', inplace=True)  # Drop rows where all elements are NaN

    fig, ax = plt.subplots(figsize=(12, 8))

    def init():
        ax.clear()
        ax.set_title(f'Comparison of {metric} over epochs', fontsize=16)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.grid(True)

    def update(frame):
        ax.clear()
        ax.set_title(f'Comparison of {metric} over epochs', fontsize=16)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.grid(True)

        for column in pivot_df.columns:
            ax.plot(pivot_df.index[:frame], pivot_df[column].iloc[:frame], label=column, linewidth=2, marker='o')
            ax.fill_between(pivot_df.index[:frame], pivot_df[column].iloc[:frame], alpha=0.1)

        ax.legend(title='Model', fontsize=12, loc='lower right')

    # Calculate frames per second
    fps = len(pivot_df) / duration
    ani = animation.FuncAnimation(fig, update, frames=len(pivot_df), init_func=init, repeat=False)

    video_path = os.path.join(save_dir, 'mAP50_95B_1.mp4')
    ani.save(video_path, writer='ffmpeg', fps=fps)
    plt.close()


"""
List all CSV files in the current directory.
"""
files = glob.glob("*.csv")

"""
Read and preprocess the CSV files.
"""
df = read_results(files)

"""
Print the columns of the dataframe to inspect the structure.
"""
print("Columns in the dataframe:", df.columns)
print(df.head())

"""
Define the metric to be compared.
"""
metric = 'metrics/mAP50-95(B)'

"""
Create the directory to save the video.
"""
save_dir = './model_videos'
os.makedirs(save_dir, exist_ok=True)

"""
Create comparison animation for the specified metric.
"""
if metric in df.columns:
    animate_comparison(df, metric, save_dir)
else:
    print(f"Metric {metric} not found in the dataframe.")
