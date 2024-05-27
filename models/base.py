from keras import layers
from keras.models import Sequential


class Model:
    def __init__(self, model_name, classes, img_height, img_width, activation='relu'):
        self.model_name = model_name
        self.num_class = len(classes)
        self.img_height = img_height
        self.img_width = img_width
        self.model = self.build_model()
        self.activation = activation

    def build_model(self):
        return Sequential([
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(self.img_height, self.img_width, 3)),

            # Augmentation
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(self.img_height, self.img_width, 3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
            layers.experimental.preprocessing.RandomContrast(0.2),

            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),

            # Regularization
            layers.Dropout(0.5),
            layers.GlobalAveragePooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_class, activation='softmax')
        ])
