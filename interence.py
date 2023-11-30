import pathlib  # path
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential

dataset_dir = pathlib.Path('datasets/mushrooms')
batch_size = 32
img_height = 180
img_width = 180

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
class_names = train_ds.class_names
print(f"Class name: {class_names}")

# cache
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create model
num_class = len(class_names)
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),

    # augmentation
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

    # regularization
    layers.Dropout(0.2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_class)
])
# compile the model
model.compile(
    optimizer='adam',
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
# model summary
model.summary()

model.load_weights('checkpoints/mushrooms/2023-11-28/weights.15-1.76.keras')
lost, acc = model.evaluate(train_ds, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# load image
url = "testing/002_pJY3-9Ttfto.jpg"
path = keras.utils.get_file('Red_sunflower', origin=url)

img = keras.utils.load_img(
    path, target_size=(img_height, img_width)
)
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# make predictions
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# print inference result
print("Looks like it's a(an) {} ({:.2f}% )".format(
    class_names[np.argmax(score)],
    100 * np.max(score)))

# show the image itself
img.show()
