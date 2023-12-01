import pathlib  # path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
import keras
from models.base import Model
import argparse

parser = argparse.ArgumentParser(description='Train a model on a dataset')
parser.add_argument("dataset", help="flowers|mushrooms, from datasets folder", type=str)
parser.add_argument("image_path", help="Epochs", type=int)
args = parser.parse_args()
DATASET_NAME = args.dataset
IMAGE_FOR_VERIFICATION = args.image_path
dataset_dir = pathlib.Path(f"datasets/{DATASET_NAME}")

batch_size = 32
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
class_names = train_ds.class_names
print(f"Class names: {class_names}")

# cache
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

num_class = len(class_names)
model = Model(model_name=DATASET_NAME, classes=class_names, img_height=img_height, img_width=img_width).build_model()
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
model.summary()

model.load_weights('runs/{}_weights'.format(DATASET_NAME))
lost, acc = model.evaluate(train_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# load image
image_path = IMAGE_FOR_VERIFICATION
img = cv2.imread(image_path)
img_resized = cv2.resize(img, (img_width, img_height))

print("Resizing image to {} x {}".format(img_width, img_height))

img_array = keras.utils.img_to_array(img_resized)
img_array = tf.expand_dims(img_array, 0)
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
print("Looks like it's a(an) {} ({:.2f}% )".format(
    class_names[np.argmax(score)],
    100 * np.max(score)))
plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
plt.show()
