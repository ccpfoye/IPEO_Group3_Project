from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np

import matplotlib.pyplot as plt


def confusion_matrix_plot(model,validation_dataset):
    # True labels and predictions
    y_true = np.concatenate([y for x, y in validation_dataset], axis=0)  
    y_pred = np.argmax(model.predict(validation_dataset), axis=1)       

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    # Customize plot (optional)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def calculate_f1_score(model,validation_dataset):
    y_true = np.concatenate([y for x, y in validation_dataset], axis=0)  
    y_pred = np.argmax(model.predict(validation_dataset), axis=1)       

    f1 = f1_score(y_true, y_pred)

    return f1


def plot_results(history,epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()