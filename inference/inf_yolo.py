# VIDEO INFERENCE
yolo task=detect mode=predict model=/Users/andreavisi/Desktop/PYTHON/"Computer Vision e Deep Learning 2024"/PROGETTO/train_w_YOLOv9c/weights/best.pt source="/Users/andreavisi/Desktop/PYTHON/Computer Vision e Deep Learning 2024/PROGETTO/Gun_Action_Recognition_Dataset/No_Gun/N8_C2_P5_V3_HB_1/video.mp4" show=True imgsz=1280 name=yolov8n_inference show_labels=True device="mps"

# LIVE INFERENCE
yolo task=detect mode=predict model=/Users/andreavisi/Desktop/PYTHON/"Computer Vision e Deep Learning 2024"/PROGETTO/train_w_YOLOv9c_2/weights/best.pt source=1 show=True imgsz=1080 name=yolov8n_inference show_labels=True

# TEST MODELLO
yolo task=detect mode=val model=/Users/andreavisi/Desktop/PYTHON/"Computer Vision e Deep Learning 2024"/PROGETTO/train_w_YOLOv8n/weights/best.pt data=guns_dataset.yaml device="mps"

