import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List

def compute_gradcam(
    model: tf.keras.Model, 
    val_dataset: tf.data.Dataset, 
    target_layer_name: str, 
    class_index: int = 0,
    resize_to_input: bool = True
) -> List[np.ndarray]:
    """
    Compute Grad-CAM heatmaps for all images in a validation dataset, 
    using the specified convolutional layer and class index.
    
    Parameters
    ----------
    model : tf.keras.Model
        A trained TF Keras model.
    val_dataset : tf.data.Dataset
        A dataset of (images, labels) or just images. 
        If (images, labels), the labels are not strictly required 
        unless you want to choose the class from the label.
    target_layer_name : str
        The name of the convolutional layer to use for Grad-CAM.
    class_index : int
        The index of the class for which to compute Grad-CAM. 
        By default, it is 0, but you might want to use:
        - The predicted class
        - A custom class index
    resize_to_input : bool
        Whether to resize the Grad-CAM heatmap to match the input image size.
        If True, the returned heatmaps will have the same HxW as the input.
        If False, they remain at the spatial size of the feature map.
    
    Returns
    -------
    heatmaps : list of np.ndarray
        A list of Grad-CAM heatmaps (2D arrays). One per image in the dataset.
        You can further normalize or overlay these on the original images.
    """

    # Create a sub-model that maps the input to (convolutional_activations, predictions)
    # We'll retrieve the outputs of the specified layer, plus the final outputs.
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(target_layer_name).output,  # feature maps
            model.output
        ]
    )

    heatmaps = []

    for batch in val_dataset:
        # If your dataset yields (images, labels)
        if isinstance(batch, (list, tuple)):
            images, labels = batch
        else:
            images = batch
        
        # Compute Grad-CAM one image at a time 
        # (you could also do this in batch form, but it's simpler to illustrate this way).
        for i in range(len(images)):
            image = images[i]

            # Add batch dimension if necessary
            input_tensor = tf.expand_dims(image, axis=0)

            with tf.GradientTape() as tape:
                # Forward pass
                conv_outputs, predictions = grad_model(input_tensor, training=False)
                
                # If you want to use the predicted class for each image, use:
                # predicted_class_index = tf.argmax(predictions[0]).numpy()
                # or for demonstration, we stick with the user-specified class_index
                loss = predictions[:, class_index]

            # Compute gradients of the selected class wrt. the feature map
            grads = tape.gradient(loss, conv_outputs)

            # Perform a global average pooling over the spatial dimensions
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Multiply each channel in the feature map by its importance weight
            conv_outputs = conv_outputs[0]  # (H, W, C) for that one image
            cam = tf.zeros(conv_outputs.shape[:2], dtype=tf.float32)

            for idx, w in enumerate(pooled_grads):
                # w is a scalar for channel idx
                cam += w * conv_outputs[:, :, idx]

            # Apply ReLU
            cam = tf.nn.relu(cam)

            # If we want to resize to the input image size
            if resize_to_input:
                # (H, W) -> (1, H, W, 1) for resizing
                cam = tf.expand_dims(cam, axis=(0, -1))
                new_size = (image.shape[0], image.shape[1])
                cam = tf.image.resize(cam, new_size, method='bilinear')
                cam = tf.squeeze(cam)  # remove batch/channel dims -> (H, W)
            
            # Convert to numpy
            cam_np = cam.numpy()

            # Optionally, normalize heatmap to [0, 1]
            if cam_np.max() > 0:
                cam_np = cam_np / cam_np.max()

            heatmaps.append(cam_np)

    return heatmaps


# Example usage:
if __name__ == "__main__":
    # Suppose you have a trained model (model) and a validation_dataset
    # usage might look like:

    # model.summary()  # to find the name of a convolutional layer
    layer_name = "conv2d_2"  # replace with the actual layer in your model
    class_idx = 0            # example class index

    # Now compute Grad-CAM heatmaps:
    gradcam_maps = compute_gradcam(model, validation_dataset, layer_name, class_idx)

    # Let's display the first one:
    if len(gradcam_maps) > 0:
        plt.imshow(gradcam_maps[0], cmap='jet')
        plt.colorbar()
        plt.title("Grad-CAM Heatmap")
        plt.show()
