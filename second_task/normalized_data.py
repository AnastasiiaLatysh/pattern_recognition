from tensorflow_core.python.debug.examples.debug_mnist import tf

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 2


def get_normalized_data():
    # load data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Normalizing images
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Normalizing labels
    train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
    test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)

    return (train_images, train_labels), (test_images, test_labels)
