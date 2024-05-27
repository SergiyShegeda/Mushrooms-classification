import os
from datetime import datetime
import argparse
import tensorflow as tf
from keras.callbacks import TensorBoard
import keras
from models.base import Model
from callbacks.checkpoint_callback import CustomModelCheckpoint
from utils import check_images, visualize

parser = argparse.ArgumentParser(description='Train a model on a dataset')
parser.add_argument("dataset", help="flowers|mushrooms, from datasets folder", type=str)
parser.add_argument("epochs", help="Epochs", type=int)
args = parser.parse_args()
DATASET_NAME = args.dataset
EPOCHS = args.epochs

print("Training model on dataset '{}'".format(DATASET_NAME))

# Ensure TensorFlow is using the Apple Silicon version
if tf.config.list_physical_devices('GPU'):
    print("Using TensorFlow for Apple Silicon")
else:
    raise RuntimeError("Please install the TensorFlow version for Apple Silicon.")

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

dataset_dir = tf.data.Dataset.list_files(f"datasets/{DATASET_NAME}/*/*")
check_images(dataset_dir)

batch_size = 32
img_height = 800
img_width = 600

train_ds = keras.utils.image_dataset_from_directory(
    directory=dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=5,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int',
)
val_ds = keras.utils.image_dataset_from_directory(
    directory=dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=5,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='int',
)

class_name = train_ds.class_names
print(f"Class names: {class_name}")

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(buffer_size=10, seed=5).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Model(model_name=DATASET_NAME, classes=class_name, img_height=img_height, img_width=img_width).build_model()
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.summary()

try:
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=[
            CustomModelCheckpoint(DATASET_NAME, save_freq=5),
            TensorBoard(log_dir="runs/{}/fit/".format(datetime.now().strftime("%Y-%m-%d")))
        ]
    )
    visualize(history, EPOCHS)
    path = "runs/{}/result".format(datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(path, exist_ok=True)

    # Save the full model
    model.save(f"{path}/{DATASET_NAME}_model", save_format="tf")
except Exception as e:
    print(f"Error during training: {e}")
