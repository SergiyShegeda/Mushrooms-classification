from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt


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


def visualize(history, epochs: int):
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
