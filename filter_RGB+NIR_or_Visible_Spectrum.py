# Filter only RGB + NIR bands
def filter_rgb_nir(images):
    return images[:, :, :, [3, 2, 1, 7]]  # Bands: Red (4), Green (3), Blue (2), NIR (8)

# Filter datasets
train_images_rgb_nir = filter_rgb_nir(train_images)
val_images_rgb_nir = filter_rgb_nir(val_images)
test_images_rgb_nir = filter_rgb_nir(test_images)

# Define the model for RGB + NIR bands
model_rgb_nir = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 12, padding='same', activation='relu', input_shape=(32, 32, 4)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 12, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 12, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)  # Binary classification
])

# Compile the model
model_rgb_nir.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Convert the filtered data to TensorFlow datasets
train_dataset_rgb_nir = tf.data.Dataset.from_tensor_slices((train_images_rgb_nir, train_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
validation_dataset_rgb_nir = tf.data.Dataset.from_tensor_slices((val_images_rgb_nir, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# Train the model
history_rgb_nir = model_rgb_nir.fit(
    train_dataset_rgb_nir,
    validation_data=validation_dataset_rgb_nir,
    epochs=10,
    class_weight=class_weights
)

# Save the model
model_rgb_nir.save('model_rgb_nir.keras')

# Filter only RGB bands
def filter_rgb(images):
    return images[:, :, :, [3, 2, 1]]  # Bands: Red (4), Green (3), Blue (2)

# Filter datasets
train_images_rgb = filter_rgb(train_images)
val_images_rgb = filter_rgb(val_images)
test_images_rgb = filter_rgb(test_images)

# Define the model for RGB bands
model_rgb = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 12, padding='same', activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 12, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 12, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)  # Binary classification
])

# Compile the model
model_rgb.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Convert the filtered data to TensorFlow datasets
train_dataset_rgb = tf.data.Dataset.from_tensor_slices((train_images_rgb, train_labels)).batch(32).prefetch(tf.data.AUTOTUNE)
validation_dataset_rgb = tf.data.Dataset.from_tensor_slices((val_images_rgb, val_labels)).batch(32).prefetch(tf.data.AUTOTUNE)

# Train the model
history_rgb = model_rgb.fit(
    train_dataset_rgb,
    validation_data=validation_dataset_rgb,
    epochs=10,
    class_weight=class_weights
)

# Save the model
model_rgb.save('model_rgb.keras')