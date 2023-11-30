import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from keras import layers
from keras.models import Sequential
import os
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train a model on a dataset')
parser.add_argument("dataset", help="flowers|mushrooms, from datasets folder", type=str)
args = parser.parse_args()
DATASET_NAME = args.dataset

print(f"Training model on dataset '{DATASET_NAME}'")


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, dataset: str = 'none', *args, **kwargs):
        self.dataset = dataset
        super().__init__(*args, **kwargs, filepath='')

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            try:
                # Get the current date and time
                current_date = datetime.now().strftime("%Y-%m-%d")

                # Create a directory path with the current date and time
                checkpoint_dir = f"checkpoints/{self.dataset}/{current_date}/"
                os.makedirs(checkpoint_dir, exist_ok=True)

                # Create a file path with placeholders for epoch and validation loss
                val_loss_str = f"{logs['val_loss']:.2f}" if 'val_loss' in logs else "unknown"
                checkpoint_path = os.path.join(checkpoint_dir, f"weights.{epoch:02d}-{val_loss_str}.keras")

                # Save the model
                self.model.save(checkpoint_path)
            except Exception as e:
                print(f"Error saving checkpoint: {e}")


dataset_dir = pathlib.Path(f"datasets/{DATASET_NAME}")
batch_size = 16
img_height = 800
img_width = 600

train_ds = keras.preprocessing.image_dataset_from_directory(
    directory=dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    directory=dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_name = train_ds.class_names
print(f"Class name: {class_name}")


# Function to check all images in the dataset
def check_images(dataset_dir):
    class_directories = list(dataset_dir.glob('*'))

    for class_dir in class_directories:
        class_name = class_dir.name
        image_paths = list(class_dir.glob('*'))

        # Use tqdm for the outer loop (class directories)
        for image_path in tqdm(image_paths, desc=f'Checking class {class_name}'):
            try:
                # Load and decode the image
                img = tf.io.read_file(str(image_path))
                img = tf.image.decode_jpeg(img, channels=3)
            except tf.errors.InvalidArgumentError as e:
                print(f"Error loading image {image_path} in class {class_name}: {e}")
            except Exception as e:
                print(f"Unexpected error loading image {image_path} in class {class_name}: {e}")


# Check images in the dataset
check_images(dataset_dir)

# Cache, shuffle, and prefetch
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Create model
num_class = len(class_name)
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

    # Augmentation
    layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
    layers.experimental.preprocessing.RandomContrast(0.2),

    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    # Regularization
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_class)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
epochs = 20
checkpoint_callback = CustomModelCheckpoint(
    dataset=DATASET_NAME,
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1)

try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint_callback])

    # Visualize training and validation results
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
    model.save_weights(f"models/{DATASET_NAME}")
except Exception as e:
    print(f"Error during training: {e}")
