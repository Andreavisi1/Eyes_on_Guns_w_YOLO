import os
import random
from ultralytics import YOLO
import cv2
import json
from datetime import datetime
from collections import Counter


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
                selected_videos[category] = "Nessun video trovato"
        else:
            selected_videos[category] = f"La cartella {category} non esiste nel percorso specificato"

    return selected_videos


def run_yolo_on_video(model, video_path, output_dir, detection_results):
    video_name = os.path.basename(video_path)
    relative_video_path = os.path.relpath(video_path, start=output_dir)
    output_path = os.path.join(output_dir, f"output_{video_name}")

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
        cv2.imshow('YOLO Detection', annotated_frame)

        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed {video_path} and saved to {output_path}")

    detection_results.append((relative_video_path, video_detection_results))


# Percorso della cartella di test
test_folder_path = '/Gun_Action_Recognition_Dataset'
output_base_folder = '/Users/andreavisi/Desktop/PYTHON/Computer Vision e Deep Learning 2024/PROGETTO/inference'

# Genera un nome unico per la cartella di output
unique_output_folder = os.path.join(output_base_folder, datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(unique_output_folder, exist_ok=True)

# Seleziona un video casuale da ciascuna categoria
random_videos = select_random_videos(test_folder_path)

# Carica il modello YOLO
model = YOLO('/epoch212/weights/best.pt')

# Esegui il modello YOLO su ciascuno dei video selezionati e salva i risultati
detection_results = []
for category, video_path in random_videos.items():
    if os.path.exists(video_path):
        category_output_folder = os.path.join(unique_output_folder, category)
        os.makedirs(category_output_folder, exist_ok=True)
        print(f"Running YOLO on video for {category}: {video_path}")
        run_yolo_on_video(model, video_path, category_output_folder, detection_results)
    else:
        print(f"Skipping {category}: {video_path}")

# Salva i risultati delle rilevazioni in un file di testo
results_file = os.path.join(unique_output_folder, 'detection_results.txt')
with open(results_file, 'w') as f:
    for relative_video_path, video_results in detection_results:
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