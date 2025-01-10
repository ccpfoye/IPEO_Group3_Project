from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np

import matplotlib.pyplot as plt


def confusion_matrix_plot(model, validation_dataset):
    """
    Return the confusion_matrix as a plot given a model and a dataset.

    Parameters
    ----------
    model : tf.keras.models.model or .sequential
        A trained TF Keras model which we can call .predict() on
    validation_dataset : tf.dataset
        A dataset of labeled images
    """
    # True labels and predictions
    y_true = np.concatenate([y for x, y in validation_dataset], axis=0)
    y_pred = np.argmax(model.predict(validation_dataset), axis=1)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    # Customize plot (optional)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def calculate_f1_score(model, validation_dataset):
    """
    Calculate the F1 score for a given model and validation dataset.
    Parameters
    ----------
    model : keras.Model
        The trained model used for making predictions.
    validation_dataset : tf.data.Dataset
        The dataset used for validation, containing features and true labels.
    Returns
    -------
    f1 : float
        The calculated F1 score.
    """
    y_true = np.concatenate([y for x, y in validation_dataset], axis=0)
    y_pred = np.argmax(model.predict(validation_dataset), axis=1)

    f1 = f1_score(y_true, y_pred)

    return f1


def plot_results(history, epochs):
    """
    Plots the training and validation accuracy and loss.
    Parameters
    ----------
    history : History
        A History object. Its `history` attribute should contain the training
        and validation accuracy and loss for each epoch.
    epochs : int
        The number of epochs.
    Returns
    -------
    None
        This function does not return anything. It displays the plots.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()
