import pathlib
import tensorflow as tf
import keras
from models.base import Model
from utils import check_images, visualize, train_model, compile_model, save_model
import argparse

parser = argparse.ArgumentParser(description='Train a model on a dataset')
parser.add_argument("dataset", help="flowers|mushrooms, from datasets folder", type=str)
parser.add_argument("epochs", help="Epochs", type=int)
args = parser.parse_args()
DATASET_NAME = args.dataset
EPOCHS = args.epochs

print("Training model on dataset '{}'".format(DATASET_NAME))

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Use CPU version if no GPU is detected
if not physical_devices:
    print("No GPU detected. Using CPU version of TensorFlow.")

dataset_dir = pathlib.Path(f"datasets/{DATASET_NAME}")
# check_images(dataset_dir)

batch_size = 32
img_height = 800
img_width = 600

train_ds = keras.preprocessing.image_dataset_from_directory(
    directory=dataset_dir,
    validation_split=0.2,
    subset="training",
    seed=100,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    directory=dataset_dir,
    validation_split=0.2,
    subset="validation",
    seed=100,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_name = train_ds.class_names
print(f"Class names: {class_name}")

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(buffer_size=200, seed=100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = Model(model_name=DATASET_NAME, classes=class_name, img_height=img_height, img_width=img_width).build_model()
compile_model(model)
model.summary()

try:
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    history = train_model(model, train_ds, val_ds, EPOCHS, steps_per_epoch, DATASET_NAME)
    visualize(history, EPOCHS)
    save_model(model, DATASET_NAME)

except Exception as e:
    print(f"Error during training: {e}")
