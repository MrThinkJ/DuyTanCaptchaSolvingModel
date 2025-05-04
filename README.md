# DuyTan Captcha Solving Model

A machine learning model designed to automatically solve captchas from the Duy Tan University portal (mydtu.duytan.edu.vn).

## Overview

This project implements a deep learning model to recognize and solve captchas from the Duy Tan University login page. The system extracts characters from captcha images, processes them, and predicts the text using a trained convolutional neural network.

## Features

- Automatic captcha recognition for Duy Tan University portal
- Image processing to extract individual characters from captchas
- Convolutional neural network for character recognition
- Flask API endpoint for seamless integration

## Project Structure

- `app.py` - Flask application that provides an API endpoint for captcha solving
- `model.py` - Definition of the CNN model architecture
- `inference.py` - Functions for model inference
- `image_processing.py` - Utility functions for captcha image processing
- `trained_model/` - Contains the trained model weights
- `data/` - Folder for storing captcha images
- `data_inference/` - Folder for storing extracted characters during inference

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Flask
- Flask-CORS
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/DuyTanCaptchaSolvingModel.git
   cd DuyTanCaptchaSolvingModel
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask server:

   ```
   python app.py
   ```

2. Send a POST request to the `/process` endpoint with the captcha image URL:

   ```
   curl -X POST -d "https://mydtu.duytan.edu.vn/Modules/ShowCaptcha.aspx" http://localhost:5000/process
   ```

3. The API will return the recognized captcha text.

## Model Architecture

The model architecture consists of:

- Convolutional neural network layers
- Batch normalization
- Max pooling
- Dropout for regularization
- Fully connected layers

The model is trained to recognize 30 different characters (digits and uppercase letters).
