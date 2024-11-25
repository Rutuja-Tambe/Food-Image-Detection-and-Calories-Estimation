Food Image Detection and Calorie Estimation
Introduction:
This project is designed to detect food items from images and estimate their calorie content. It uses machine learning models,
including  CNN for classification and calorie estimation. The goal is to provide an easy-to-use tool for health and nutrition monitoring by analyzing food images.

Features:
Detects food items in images using object detection.
Classifies detected food items into predefined categories.
Estimates the calorie content of the detected food items.

The project contains the following key files and folders:

Files:

CNN.ipynb: Jupyter notebook for training and evaluating the CNN model.

app1.py: The main Python script for running the application.

features.ipynb: Notebook for extracting and analyzing features from the dataset.

food calories estimation from image ppt.pptx: Presentation detailing the project overview.

label_encoder.pkl: Saved label encoder for mapping class labels.

training_history.json: Training history for the CNN model.

result.ipynb, result1.ipynb: Notebooks for testing and visualizing model predictions.

.gitignore: Specifies intentionally untracked files to ignore.

Folders:

dataset/: Contains images of various food items organized by class.

features/: Stores extracted features from images.

static/uploads/: Directory for storing uploaded images during app runtime.

templates/: HTML templates for the web interface.

test/: Contains test cases and sample input images.

verify/: Utility scripts or additional files for verification and validation.

Dataset

The dataset contains images of 26 classes of food items. Images are stored in separate folders, each representing a food category. 
These images are used for training, validation, and testing.
