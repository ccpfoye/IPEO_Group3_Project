import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import glob

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

################################################################################
# If you have a custom module with plotting functions:
# from scr.plot import confusion_matrix_plot, calculate_f1_score, plot_results
# For demonstration, here are placeholder definitions:
def confusion_matrix_plot(model, dataset):
    """Example placeholder function. Replace with your actual implementation."""
    y_true = []
    y_pred = []
    for x, y in dataset:
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(model.predict(x), axis=1))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def calculate_f1_score(model, dataset):
    """Example placeholder function. Replace with your actual implementation."""
    from sklearn.metrics import f1_score
    y_true = []
    y_pred = []
    for x, y in dataset:
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(model.predict(x), axis=1))
    return f1_score(y_true, y_pred, average='macro')

def plot_results(history, epochs):
    """Example placeholder function. Replace with your actual implementation."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(14,5))
    plt.subplot(1,2,1)
    plt.plot(range(epochs), acc, label='Training Accuracy')
    plt.plot(range(epochs), val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1,2,2)
    plt.plot(range(epochs), loss, label='Training Loss')
    plt.plot(range(epochs), val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()
################################################################################

print("TensorFlow version:", tf.__version__)

def load_data_subset_from_directory(directory, max_files=None):
    """
    Load data from only a subset of files in a directory to save memory.
    
    Parameters:
    -----------
    directory : str
        Path to directory containing pickled files.
    max_files : int or None
        If an integer, limit the number of files to load to this many.
        If None, load all files.
    
    Returns:
    --------
    (images, labels) : tuple of np.ndarrays
    """
    all_files = os.listdir(directory)
    # Ensure a consistent ordering or random sampling as needed
    all_files.sort()  # or use random.shuffle(all_files) if you want a random subset

    # If max_files is set, truncate the list of files
    if max_files is not None and max_files < len(all_files):
        all_files = all_files[:max_files]

    images = []
    labels = []
    for filename in all_files:
        filepath = os.path.join(directory, filename)
        with open(filepath, 'rb') as f:
            image, label = pickle.load(f)
            # Reshape or reorder image data if necessary (bands, height, width -> height, width, bands)
            image = np.transpose(image, (1, 2, 0))
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# Paths to your data directories
train_dir = "data/classification_dataset/train"
validation_dir = "data/classification_dataset/validation"

# Load the train data (you can load the full set or also limit it)
train_images, train_labels = load_data_subset_from_directory(train_dir, max_files=10)  

# Load ONLY a subset of validation data, e.g. the first 10 files
val_images, val_labels = load_data_subset_from_directory(validation_dir, max_files=10)

# Convert data to tf.data.Dataset
def create_tf_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset

train_dataset = create_tf_dataset(train_images, train_labels)
validation_dataset = create_tf_dataset(val_images, val_labels)

# Batch and shuffle the datasets
BATCH_SIZE = 32
SEED = 42

train_dataset = (
    train_dataset
    .shuffle(buffer_size=1000, seed=SEED)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Train set size:", len(train_images))
print("Validation set size (subset):", len(val_images))

# Quick sanity check
for images_batch, labels_batch in train_dataset.take(1):
    print("Shapes in one train batch:", images_batch.shape, labels_batch.shape)

# Display one sample image from the train set
for images_batch, labels_batch in train_dataset.take(1):
    image = images_batch[0]
    label = labels_batch[0]

    # Example: pick bands 4,3,2 if indices are 3,2,1
    rgb_image = np.stack([
        image[:, :, 3],
        image[:, :, 2],
        image[:, :, 1]
    ], axis=-1)

    # Normalize the RGB image for displaying
    rgb_image = (rgb_image / np.max(rgb_image) * 255).astype(np.uint8)
    plt.imshow(rgb_image)
    plt.title(f"Label: {label.numpy()}")
    plt.axis("off")
    plt.show()
    break
