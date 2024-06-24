import os
import random
from ultralytics import YOLO
import cv2
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import json

"""
Select a random video from the specified folder.
"""
def select_random_video_from_folder(folder_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))

    if not video_files:
        return None

    random_video = random.choice(video_files)
    return random_video

"""
Select random videos from the base folder for specified categories.
"""
def select_random_videos(base_folder):
    categories = ['Handgun', 'Machine_Gun', 'No_Gun']
    selected_videos = {}

    for category in categories:
        category_path = os.path.join(base_folder, category)
        if os.path.isdir(category_path):
            random_video = select_random_video_from_folder(category_path)
            if random_video:
                selected_videos[category] = random_video
            else:
                selected_videos[category] = "No video found"
        else:
            selected_videos[category] = f"The folder {category} does not exist in the specified path"

    return selected_videos

"""
Run YOLO model on the specified video and save the detection results.
"""
def run_yolo_on_video(model, video_path, output_dir, detection_results, model_name):
    video_name = os.path.basename(video_path)
    relative_video_path = os.path.relpath(video_path, start=output_dir)
    output_path = os.path.join(output_dir, f"output_{model_name}_{video_name}")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    video_detection_results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            annotated_frame = result.plot()  # Get annotated frame
            if result.boxes:
                for box in result.boxes:
                    if box.cls is not None and box.conf is not None:
                        cls_name = model.names[int(box.cls)]
                        video_detection_results.append({
                            "frame": frame_number,
                            "class": cls_name,
                            "confidence": box.conf.item()
                        })

        out.write(annotated_frame)

        # Display the frame
        cv2.imshow(f'YOLO Detection ({model_name})', annotated_frame)

        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed {video_path} with {model_name} and saved to {output_path}")

    detection_results[model_name].append((relative_video_path, video_detection_results))

"""
Parse the detection results from the specified file.
"""
def parse_detection_results(file_path):
    detection_counts = defaultdict(lambda: defaultdict(int))
    current_model = None
    video_detections = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Model:"):
                current_model = line.split(": ")[1]
            elif line.startswith("Path:"):
                current_path = line.split(": ")[1]
            elif line.startswith("Video:"):
                current_video = line.split(": ")[1]
            elif line.startswith("Class '"):
                class_name = line.split("'")[1]
                count = int(line.split(": ")[1])
                if class_name != "No_Gun":
                    detection_counts[current_model][class_name] += count
                    video_detections[current_model][current_video].append((class_name, count))

    return detection_counts, video_detections

"""
Load expected detections from the specified JSON file.
"""
def load_expected_detections(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    expected_detections = defaultdict(lambda: defaultdict(int))
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        video_name = next(img['file_name'] for img in data['images'] if img['id'] == image_id)
        category_name = next(cat['name'] for cat in data['categories'] if cat['id'] == category_id)
        expected_detections[video_name][category_name] += 1

    return expected_detections

# Path to the test folder
test_folder_path = '/Users/andreavisi/Desktop/PYTHON/Computer Vision and Deep Learning 2024/PROJECT/Gun_Action_Recognition_Dataset'
output_base_folder = '/Users/andreavisi/Desktop/PYTHON/Computer Vision and Deep Learning 2024/PROJECT/inference'

# Generate a unique name for the output folder
unique_output_folder = os.path.join(output_base_folder, datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(unique_output_folder, exist_ok=True)

# Select a random video from each category
random_videos = select_random_videos(test_folder_path)

# Define the paths to the models
model_paths = {
    #"YOLOv8n": "/Users/andreavisi/Desktop/PYTHON/Computer Vision and Deep Learning 2024/PROJECT/E1 - train_w_YOLOv8n/weights/best.pt",
    "YOLOv8m": "/Users/andreavisi/Desktop/PYTHON/Computer Vision and Deep Learning 2024/PROJECT/E3 - train_w_YOLOv8m/weights/best.pt",
    "YOLOv9c": "/Users/andreavisi/Desktop/PYTHON/Computer Vision and Deep Learning 2024/PROJECT/E2 - train_w_YOLOv9c/weights/best.pt",
    "YOLOv9c_2": "/Users/andreavisi/Desktop/PYTHON/Computer Vision and Deep Learning 2024/PROJECT/E4 - train_w_YOLOv9c_2/weights/best.pt",
    #"YOLOv10b": "/Users/andreavisi/Desktop/PYTHON/Computer Vision and Deep Learning 2024/PROJECT/E5 - train_w_YOLOv10b/weights/best.pt"
}

# Load YOLO models
models = {name: YOLO(path) for name, path in model_paths.items()}

# Run YOLO model on each selected video and save the results
detection_results = defaultdict(list)
for model_name, model in models.items():
    for category, video_path in random_videos.items():
        if os.path.exists(video_path):
            category_output_folder = os.path.join(unique_output_folder, category)
            os.makedirs(category_output_folder, exist_ok=True)
            print(f"Running YOLO ({model_name}) on video for {category}: {video_path}")
            run_yolo_on_video(model, video_path, category_output_folder, detection_results, model_name)
        else:
            print(f"Skipping {category}: {video_path}")

# Save detection results to a text file
results_file = os.path.join(unique_output_folder, 'detection_results.txt')
with open(results_file, 'w') as f:
    for model_name, videos in detection_results.items():
        f.write(f"Model: {model_name}\n")
        for relative_video_path, video_results in videos:
            category = os.path.basename(os.path.dirname(relative_video_path))
            f.write(f"Path: {category}\n")
            f.write(f"Video: {relative_video_path}\n")
            frame_counts = Counter([result['class'] for result in video_results])
            for result in video_results:
                f.write(f"Frame: {result['frame']}, Class: {result['class']}, Confidence: {result['confidence']:.2f}\n")
            f.write(f"Total detected frames: {len(video_results)}\n")
            for cls_name, count in frame_counts.items():
                f.write(f"Class '{cls_name}' detected frames: {count}\n")
            f.write("\n")

print(f"All results saved to {unique_output_folder}")

# Path to the results file
results_file_path = os.path.join(unique_output_folder, 'detection_results.txt')

# Read and analyze the results
detection_counts, video_detections = parse_detection_results(results_file_path)

# Prepare data for the plot
categories = ['Handgun', 'Machine_Gun', 'No_Gun']
x = np.arange(len(categories))  # category indices
width = 0.15  # bar width

# Generate a set of similar colors
colors = {
    "YOLOv8n": {"Handgun": "#1f77b4", "Machine_Gun": "#aec7e8"},
    "YOLOv8m": {"Handgun": "#ff7f0e", "Machine_Gun": "#ffbb78"},
    "YOLOv9c": {"Handgun": "#2ca02c", "Machine_Gun": "#98df8a"},
    "YOLOv9c_2": {"Handgun": "#d62728", "Machine_Gun": "#ff9896"},
    "YOLOv10b": {"Handgun": "#9467bd", "Machine_Gun": "#c5b0d5"}
}
expected_color = '#808080'  # Gray for expected detections

fig, ax = plt.subplots(figsize=(12, 8))  # Increase figure size

# Plot bars for each model and class
for i, (model_name, counts) in enumerate(detection_counts.items()):
    for j, category in enumerate(categories):
        if category != "No_Gun":
            count = counts.get(category, 0)
            bar_x = x[j] + i * width  # bar position
            ax.bar(bar_x, count, width, label=f'{model_name} ({category})', color=colors[model_name][category])
            ax.annotate(f'{count}', xy=(bar_x, count), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Handle detections in No_Gun videos
for i, (model_name, videos) in enumerate(video_detections.items()):
    for video_path, detections in videos.items():
        if 'No_Gun' in video_path:
            for detection in detections:
                class_name, count = detection
                index = categories.index("No_Gun")
                bar_x = x[index] + list(model_paths.keys()).index(model_name) * width
                ax.bar(bar_x, count, width, label=f'{model_name} ({class_name})', color=colors[model_name][class_name])
                ax.annotate(f'{count}', xy=(bar_x, count), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Handle multiple detections in a single video
for model_name, videos in video_detections.items():
    for video_path, detections in videos.items():
        for detection in detections:
            class_name, count = detection
            category = os.path.basename(os.path.dirname(video_path))
            if category != 'No_Gun' and category in categories:
                bar_x = x[categories.index(category)] + list(model_paths.keys()).index(model_name) * width
                ax.bar(bar_x, count, width, label=f'{model_name} ({class_name})', color=colors[model_name][class_name])
                ax.annotate(f'{count}', xy=(bar_x, count), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Add gray bars for expected detections and dashed horizontal lines
for category, video_path in random_videos.items():
    if category != "No_Gun":
        json_path = os.path.join(os.path.dirname(video_path), 'label.json')
        if os.path.exists(json_path):
            expected_detections = load_expected_detections(json_path)
            for video_name, class_counts in expected_detections.items():
                for class_name, expected_count in class_counts.items():
                    if class_name in categories:
                        bar_x = x[categories.index(category)] - width  # position of the expected detection bar
                        ax.bar(bar_x, expected_count, width, label=f'Expected ({class_name})', color=expected_color)
                        ax.annotate(f'{expected_count}', xy=(bar_x, expected_count), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                        # Add the dashed horizontal line
                        ax.axhline(y=expected_count, color=expected_color, linestyle='--')

# Print the number of detections for each weapon type for each video
for model_name, videos in video_detections.items():
    for video_path, detections in videos.items():
        print(f"Model: {model_name}, Video: {video_path}")
        for detection in detections:
            print(f"Class: {detection[0]}, Count: {detection[1]}")
        print()

# Add labels and legend
ax.set_xlabel('Video Category')
ax.set_ylabel('Number of Detections')
ax.set_title('Number of Detections by Model and Video Category')
ax.set_xticks(x + width / 2 * (len(model_paths) - 1))
ax.set_xticklabels(categories)
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right', ncol=2, fontsize='small')  # Position legend in upper right

# Save the plot
plot_file_path = os.path.join(unique_output_folder, 'detection_results_plot.png')
plt.savefig(plot_file_path)
plt.show()
