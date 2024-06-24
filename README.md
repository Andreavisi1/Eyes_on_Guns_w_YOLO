# Gun Action Recognition with YOLO and EigenCAM

This repository contains a project focused on training custom YOLO models for gun action recognition and utilizing EigenCAM for explainable AI evaluation of the results. The project includes data preparation, training, and visualization steps to ensure accurate and interpretable results.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Using EigenCAM for Explainable AI](#using-eigencam-for-explainable-ai)
- [Results and Evaluation](#results-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to recognize gun actions using custom-trained YOLO models and to evaluate the results using EigenCAM for explainable AI. YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. EigenCAM is used to provide visual explanations for the model's decisions, making the results more interpretable.

## Installation

To get started, clone this repository and install the necessary dependencies:

```bash
git clone [https://github.com/Andreavisi1/Eyes_on_Guns_w_YOLO]
cd gun-action-recognition
```

## Dataset Preparation

The dataset consists of videos categorized into `Handgun`, `Machine_Gun`, and `No_Gun`. Each category contains videos from which frames are extracted and labeled. The dataset structure should be as follows:

```
Gun_Action_Recognition_Dataset/
|
|--- Handgun/
|--- Machine_Gun/
|--- No_Gun/
```

### Extract Frames and Convert Labels

Run the script to extract frames from videos and convert labels to YOLO format:

```python
# Extract frames and convert labels
python extract_and_convert.py
```

## Training the Model

Custom YOLO models can be trained using the provided scripts. The models are trained on the prepared dataset with specified configurations.

### Example Training Command

```python
# Initialize YOLO model
model = YOLO('basic_models/choose_favourite_model.pt')

# Custom training function
def custom_train(model, yaml_filename, epochs=10, batch_size=16, imgsz=480):
    # Training logic here...

# Execute custom training
custom_train(model, 'choose_file_yaml.yaml', epochs=10, batch_size=16, imgsz=640)
```

## Using EigenCAM for Explainable AI

EigenCAM is used to generate visual explanations for the YOLO model's predictions. This helps in understanding why the model makes certain predictions.

### Example Usage of EigenCAM

```python
from yolov8_cam.eigen_cam import EigenCAM
from yolov8_cam.utils.image import show_cam_on_image

# Load the trained YOLO model
model = YOLO('choose_favourite_model.pt')

# Use EigenCAM for explanations
cam = EigenCAM(model, target_layers, task='od')
grayscale_cam = cam(rgb_img, eigen_smooth=True, principal_comp=principal_comp)

for i in range(grayscale_cam.shape[3]):
    cam_image = show_cam_on_image(img, grayscale_cam[0,:,:,i], use_rgb=True)
    plt.imshow(cam_image)
    plt.show()
```

## Results and Evaluation

After training, evaluate the model using the test set and visualize the results using EigenCAM. The explanations can be saved for further analysis.

### Example Evaluation Command

```python
# Verify the dataset structure
if check_dataset_structure(dataset_base_path):
    print("Dataset structure is correct.")
else:
    print("Dataset structure has issues.")
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
