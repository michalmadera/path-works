import tensorflow as tf
import keras
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from keras import layers


def check_GPU():
    print(tf.config.list_physical_devices('GPU'))

def get_dataset(batch_size, img_size, input_img_paths, target_img_paths, max_dataset_len=None,
                ):
    def load_img_masks(input_img_path, target_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        return input_img, target_img
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # x = layers.UpSampling2D(2)(x)
        x = layers.UpSampling2D(2, interpolation='nearest')(x)

        # Project residual
        residual = layers.UpSampling2D(2, interpolation='nearest')(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        # residual = layers.UpSampling2D(2)(previous_block_activation)
        # residual = layers.Conv2D(filters, 1, padding="same")(residual)

        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


def compile_model(model, optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy"):
    model.compile(
        optimizer=optimizer, loss=loss
    )


def save_model(save_name="oxford_segmentation.tf", save_best_only=True):
    callbacks = [
        keras.callbacks.ModelCheckpoint(save_name, save_best_only=save_best_only)
    ]
    return callbacks


def train_model(model, train_dataset, validation_dateset, callbacks, epochs=50, verbose=2):
    model.fit(train_dataset, epochs, validation_dateset, callbacks, verbose)


def model_prediction(model, val_dataset):
    prediction = model.predict(val_dataset)
    return prediction

def load_model(save_name = 'oxford_segmentation.tf'):
    model = tf.keras.models.load_model(save_name)
    return model