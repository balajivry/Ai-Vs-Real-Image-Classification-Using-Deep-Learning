# Ai-Vs-Real-Image-Classification-Using-Deep-Learning

## Overview
A web application to detect whether an image is AI-generated or Real Image using Machine Learning. This project features an intuitive web interface built with Flask, allowing users to upload an image and receive a prediction. The application structure includes app.py for the Flask app,in the main folder create a sub folder templates for web page home.html , and a static directory for uploaded images. To get started, install requirements, ensure the static and templates directories exist, and place home.html in templates. Run the Flask app locally with python app.py and access the web interface at http://127.0.0.1:5000/. Make sure to rename the images by using the image renaming script, place all the files and folders in one main folder.

### Dataset

You can find the dataset we used for this project on Kaggle: [Fake AI-generated and Real Image Dataset](https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images/data?select=train).

## Project Structure

The project consists of the following main components:
- `app.py/`: Here, you can find the overall application code to start with
- `codd.ipynb`: A Jupyter Notebook explaining the model training process.
- `best_model.keras`: The file best_model.keras contains a saved version of the trained model, which prevents the need to retrain the model each time it is tested.
- `home.html`: The file home.html consists of combined html and css scripts for the web site.

## Directory Structure
    ## Main folder
    |-- train_data                              # Folder consists of 2 folders real and fake each consists of 24000 images total of 48000 images.
    |-- test_data                               # Folder consists of consists of 2 folder real and fake each consists of 6000 images total of 12000 images
    |-- Templates Folder------> home.html       # Folder for HTML templates keep home.html in templates folder
    |-- Static Folder                           # Folder for static files and uploaded images
    |-- app.py                                  # Main application file
    |-- best_model.keras                        # best_model.keras file for the trained model
    |-- codd.ipynb                              # Jupyter Notebook for model training


## Model Development

I have used a MobilenetV2 for image classification. The model was trained on a labeled dataset containing both AI-generated and Real images. 
