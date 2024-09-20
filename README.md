# Road Crack Detection using Google Street View Images

## Overview

This code is designed for road crack detection using Google Street View images. It includes the following functionalities:

1. **Data Collection**: Utilizes the Google Street View API to gather images at different locations along specified coordinates.

2. **Data Processing**: Extracts metadata from the downloaded images and organizes them into a structured dataset.

3. **Data Transformation**: Converts the collected data into a CSV file with relevant information, including location details and image filenames.

4. **Image Labeling**: Labels the images with crack or no-crack classes and updates the dataset accordingly.

5. **Model Development**: Implements a VGG16-based Convolutional Neural Network (CNN) for road crack detection.

6. **Training**: Trains the CNN model on the labeled dataset to learn road crack patterns.

7. **Testing**: Evaluates the model on a separate test set to measure its accuracy.

8. **Localization Visualization**: Generates visualizations with randomly generated bounding box localizations on images for evaluation.

## Requirements

- Python 3.x
- Google Street View API key
- PyTorch
- Matplotlib
- Pandas
- Requests
- Google Colab

## Usage

1. Install required packages:

    ```bash
    !pip install google-streetview
    ```

2. Mount Google Drive:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

3. Update Google Street View API key in the code.

4. Execute the code step by step.

5. Model training will be performed, and accuracy will be displayed after testing.

## Folder Structure

- `/content/drive/MyDrive/Road Crack Detection/`: Directory for storing downloaded Google Street View images.
- `/content/drive/MyDrive/Sample Data/6817 Calumet Ave/`: Example directory with metadata and labeled images.
- `/content/drive/MyDrive/Image Dataset/`: Main directory for storing the image dataset.
- `/content/drive/MyDrive/Image Dataset/Cracked/`: Subdirectories for storing cracked images based on labeling.

## License

This project is licensed under the [MIT License](https://opensource.org/license/mit/)
