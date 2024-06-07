import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """

    images = []
    labels = []

    # Loop over each category directory
    for category in range(NUM_CATEGORIES):
        category_dir = os.path.join(data_dir, str(category))
        
        # Check if category_dir is a directory
        if os.path.isdir(category_dir):
            # Loop over each image file in the category directory
            for file in os.listdir(category_dir):
                file_path = os.path.join(category_dir, file)
                
                # Load the image
                image = cv2.imread(file_path)
                if image is not None:
                    # Resize the image to the desired dimensions
                    resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    # Append the image and its label to the lists
                    images.append(resized_image)
                    labels.append(category)
                else:
                    print(f"Warning: Failed to load image {file_path}")

    return images, labels

    raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """ 
    # Define a Sequential model
    model = models.Sequential([
        # First convolutional layer with 32 filters of size 3x3, ReLU activation, and specified input shape
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        # First max pooling layer with a 2x2 window
        layers.MaxPooling2D((2, 2)),
        # Second convolutional layer with 64 filters of size 3x3, ReLU activation
        layers.Conv2D(64, (3, 3), activation='relu'),
        # Second max pooling layer with a 2x2 window
        layers.MaxPooling2D((2, 2)),
        # Flatten layer to convert 3D output to 1D
        layers.Flatten(),
        # Fully connected dense layer with 128 neurons and ReLU activation
        layers.Dense(128, activation="relu"),
        # Dropout layer to prevent overfitting with a dropout rate of 0.2
        layers.Dropout(0.2),
        # Fully connected dense layer with 256 neurons and ReLU activation
        layers.Dense(256, activation='relu'),
        # Dropout layer to prevent overfitting with a dropout rate of 0.2
        layers.Dropout(0.2),
        # Output layer with number of neurons equal to the number of categories, softmax activation for classification
        layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model with Adam optimizer, categorical crossentropy loss, and accuracy metric
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
    raise NotImplementedError


if __name__ == "__main__":
    main()