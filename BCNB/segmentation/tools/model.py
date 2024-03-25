import os.path

import tensorflow as tf
import keras
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from keras import layers
from tensorflow.keras import backend as K

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


def compile_model(model, optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=['accuracy']):
    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics
    )


def make_callbacks(save_path="", save_best_only=True, log_dir="logs/fit"):
    os.makedirs(log_dir, exist_ok=True)
    folder_count = len([name for name in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, name))])
    log_dir = os.path.join(log_dir, f"test_{folder_count}")
    save_name = os.path.join(save_path, f"model_{folder_count}.tf")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)
    save_callback = keras.callbacks.ModelCheckpoint(save_name, save_best_only=save_best_only)
    callbacks = [tensorboard_callback, save_callback]
    return callbacks


def train_model(model, train_dataset, validation_dataset, callbacks, epochs=50, verbose=2):
    model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, callbacks=callbacks, verbose=verbose)


def model_prediction(model, val_dataset):
    prediction = model.predict(val_dataset)
    return prediction

def load_model(save_name = 'oxford_segmentation.tf'):
    model = tf.keras.models.load_model(save_name)
    return model

def dice_coef(y_true, y_pred, smooth=100):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice.numpy()

def AUROC(y_true, y_pred):
    y_true_tensor = tf.constant(y_true)
    y_pred_tensor = tf.constant(y_pred)

    auc_metric = tf.keras.metrics.AUC()

    auc_metric.update_state(y_true_tensor, y_pred_tensor)

    auroc_value = auc_metric.result()

    return auroc_value.numpy()

def ACC(y_true, y_pred):
    y_true_tensor = tf.constant(y_true)
    y_pred_tensor = tf.constant(y_pred)

    acc_metric = tf.keras.metrics.Accuracy()

    acc_metric.update_state(y_true_tensor, y_pred_tensor)

    acc_value = acc_metric.result()

    return acc_value.numpy()

def IoU(y_true, y_pred, num_classes = 2, target_class_ids=[0]):
    y_true_tensor = tf.constant(y_true)
    y_pred_tensor = tf.constant(y_pred)

    IoU_metric = tf.keras.metrics.IoU(num_classes, target_class_ids)

    IoU_metric.update_state(y_true_tensor, y_pred_tensor)

    IoU_value = IoU_metric.result()

    return IoU_value.numpy()
def calculate_iou_and_dice(y_true ,y_pred):
    auroc_value = AUROC(y_true, y_pred)
    acc_value = ACC(y_true, y_pred)
    dice = dice_coef(y_true, y_pred)
    iou = IoU(y_true, y_pred)

    return acc_value, auroc_value, iou, dice

if __name__ == '__main__':
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 0.3, 0.2, 0.3, 0.7]
    acc, auroc, iou, dice = calculate_iou_and_dice(y_true, y_pred)
    print(f"ACC: {acc}")
    print(f"AUROC: {auroc}")
    print(f"IoU: {iou}")
    print(f"Dice: {dice}")