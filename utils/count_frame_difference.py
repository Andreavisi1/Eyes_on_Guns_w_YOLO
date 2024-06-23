import os
import json

def count_frames_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        image_ids = {item['id'] for item in data['images']}
        return len(image_ids)

def count_images_in_labels(labels_dir):
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_count = 0

    for root, _, files in os.walk(labels_dir):
        image_count += sum(1 for file in files if os.path.splitext(file)[1].lower() in image_extensions)

    return image_count

def process_directory(directory):
    total_images = 0
    total_frames = 0
    subfolder_image_counts = {}
    subfolder_frame_counts = {}

    for root, dirs, files in os.walk(directory):
        if 'label.json' in files:
            json_path = os.path.join(root, 'label.json')
            frame_count = count_frames_from_json(json_path)
            subfolder_frame_counts[root] = frame_count
            total_frames += frame_count

        if 'frames' in dirs:
            labels_dir = os.path.join(root, 'frames')
            image_count = count_images_in_labels(labels_dir)
            subfolder_image_counts[labels_dir] = image_count
            total_images += image_count

    return total_images, subfolder_image_counts, total_frames, subfolder_frame_counts

directories = ['/Users/andreavisi/Desktop/PYTHON/Computer Vision e Deep Learning 2024/PROGETTO/Gun_Action_Recognition_Dataset_Frames/Handgun',
        '/Users/andreavisi/Desktop/PYTHON/Computer Vision e Deep Learning 2024/PROGETTO/Gun_Action_Recognition_Dataset_Frames/Machine_Gun',
        '/Users/andreavisi/Desktop/PYTHON/Computer Vision e Deep Learning 2024/PROGETTO/Gun_Action_Recognition_Dataset_Frames/No_Gun']

for directory in directories:
    total_images, subfolder_image_counts, total_frames, subfolder_frame_counts = process_directory(directory)
    print(f'Directory: {directory}')
    print(f'Frames estratti: {total_images}')
    print(f'Frames presenti nel file label.json: {total_frames}')
    print(f'Differenza: {total_images - total_frames}')

