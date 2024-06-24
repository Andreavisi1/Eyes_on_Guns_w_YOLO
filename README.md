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
git clone https://github.com/your-username/gun-action-recognition.git
cd gun-action-recognition
pip install -r requirements.txt
```

## Dataset Preparation

The dataset consists of videos categorized into `Handgun`, `Machine_Gun`, and `No_Gun`. Each category contains videos from which frames are extracted and labeled. The dataset structure should be as follows:

```
Gun_Action_Recognition_Dataset/
    Handgun/
    Machine_Gun/
    No_Gun/
```

### Extract Frames and Convert Labels

Run the script to extract frames from videos and convert labels to YOLO format:

```bash
python extract_and_convert.py
```

## Training the Model

Custom YOLO models can be trained using the provided scripts. The models are trained on the prepared dataset with specified configurations.

### Example Training Command

Initialize the YOLO model with a pre-trained weight file and run the custom training function provided in the repository. The training script handles error checking and saves problematic batches for further inspection if any errors occur.

## Using EigenCAM for Explainable AI

EigenCAM is used to generate visual explanations for the YOLO model's predictions. This helps in understanding why the model makes certain predictions.

### Example Usage of EigenCAM

Load the trained YOLO model, use EigenCAM for explanations, and visualize the results. This process will generate and display CAM images that highlight the areas in the input images that contributed most to the model's predictions.

## Results and Evaluation

After training, evaluate the model using the test set and visualize the results using EigenCAM. The explanations can be saved for further analysis.

### Example Evaluation Command

Run the provided script to verify the dataset structure and ensure that the data splits are correctly assigned. This will help in validating the training and testing data setup before running the model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Copy and paste this README into your GitHub repository to provide clear and comprehensive instructions for using the project.
