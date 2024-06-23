# 1. Import necessary libraries
from ultralytics import YOLO # Here we import YOLO
import yaml                  # for yaml files
import torch
from PIL import Image
import os
import cv2
import time

# 2. Choose our yaml file
yaml_filename = '../guns_dataset.yaml'

# 3. Create Yolo model
model = YOLO('yolov8n.yaml') # creates Yolo object from 'yolov8n.yaml' configuration file.
model = YOLO('Gun_Action_Recognition_Dataset_Frames/yolov8n.pt')   # Loads pretrained weights
model = YOLO('yolov8n.yaml').load('Gun_Action_Recognition_Dataset_Frames/yolov8n.pt')  # build from YAML and transfer weights

# 4. Train the model
model.train(data='{}'.format(yaml_filename), epochs=30,patience=5, batch=16,  imgsz=640)

# 5. Load the trained weights
model = YOLO('runs/train/weights/best.pt')

# 6. to run prediction on 1 image at a time
im = Image.open('new_car_images/car99.jpg')
results = model.predict(source=im, save=True)

# 7. to run prediction on all images in a folder
im_dir = 'new_car_images/'
results = model.predict(source=im_dir, save=True)