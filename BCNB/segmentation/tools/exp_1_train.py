import keras
import model as mod
from tensorflow.keras.callbacks import TensorBoard
import cv2_visualise as cv2_visualizer
import re
import os
import numpy as np
from PIL import ImageOps

batch_size = 32
epochs = 50
img_size = (256, 256)
num_classes = 2

train_input_img_paths_dir = 'data/train-image-tiles/'
train_target_img_paths_dir = 'data/train-mask-tiles/'

train_input_img_paths = sorted([os.path.join(train_input_img_paths_dir, fname) for fname in os.listdir(train_input_img_paths_dir)])
train_target_img_paths = sorted([os.path.join(train_target_img_paths_dir, fname) for fname in os.listdir(train_target_img_paths_dir)])

val_input_img_paths_dir = 'data/valid-image-tiles/'
val_target_img_paths_dir = 'data/valid-mask-tiles/'

val_input_img_paths = sorted([os.path.join(val_input_img_paths_dir, fname) for fname in os.listdir(val_input_img_paths_dir)])
val_target_img_paths = sorted([os.path.join(val_target_img_paths_dir, fname) for fname in os.listdir(val_target_img_paths_dir)])

test_input_img_paths_dir = 'data/test-image-tiles/'
test_target_img_paths_dir = 'data/test-mask-tiles/'

test_input_img_paths = sorted([os.path.join(test_input_img_paths_dir, fname) for fname in os.listdir(test_input_img_paths_dir)])
test_target_img_paths = sorted([os.path.join(test_target_img_paths_dir, fname) for fname in os.listdir(test_target_img_paths_dir)])
test_input_img_paths = sorted(test_input_img_paths, key=lambda x: [int(i) if i.isdigit() else i for i in re.split('(\d+)', x)])
test_target_img_paths = sorted(test_target_img_paths, key=lambda x: [int(i) if i.isdigit() else i for i in re.split('(\d+)', x)])

train_dataset = mod.get_dataset(batch_size, img_size,
                                train_input_img_paths, train_target_img_paths)

valid_dataset = mod.get_dataset(batch_size, img_size,
                                val_input_img_paths, val_target_img_paths)

model = mod.get_model(img_size=img_size, num_classes=num_classes)

optimizer = keras.optimizers.Adam(1e-8)
mod.compile_model(model, optimizer=optimizer)

# TODO: Handle tensorboard callback in a better way. Use automatic folder naming.
callbacks = mod.make_callbacks()

mod.train_model(model, train_dataset=train_dataset, validation_dataset=valid_dataset, epochs=epochs, callbacks=callbacks, verbose=2)

# Test the model
test_dataset = mod.get_dataset(batch_size, img_size,
                               test_input_img_paths, test_target_img_paths)

# Open the best persisted model
#model = mod.load_model()

# Predict on the test dataset
predictions = mod.model_prediction(model, valid_dataset)

# TODO: Implement mehtod to calculate ACC, AUROC, IoU, and Dice for the predictions
acc, auroc, iou, dice = mod.calculate_iou_and_dice(test_dataset, predictions)

# Print the results
print(f"ACC: {acc}")
print(f"AUROC: {auroc}")
print(f"IoU: {iou}")
print(f"Dice: {dice}")

#Visualize the predictions for the first 4 images
# TODO: Use visualization methods you prepared of the predictions. Save the visualizations in 4 files.
number_of_images_to_vis = 4

def find_images_with_prefix(base_path, prefix):
    image_set = set()
    for fname in os.listdir(base_path):
        if fname.startswith(prefix):
            image_set.add(os.path.join(base_path, fname))
    sorted_images = sorted(image_set, key=lambda x: [int(i) if i.isdigit() else i for i in re.split('(\d+)', x)])
    return sorted_images

def generate_pred_list(i, prediction):
    mask = np.argmax(prediction[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.utils.array_to_img(mask))
    test_pred_list.append(img)
val_input_img_paths = sorted([os.path.join("data/valid-images", fname) for fname in os.listdir("data/valid-images")])
val_target_img_paths = sorted([os.path.join("data/valid-masks", fname) for fname in os.listdir("data/valid-masks")])

train_input_img_paths = sorted([os.path.join("data/train-images", fname) for fname in os.listdir("data/train-images")])
train_target_img_paths = sorted([os.path.join("data/train-masks", fname) for fname in os.listdir("data/train-masks")])

len_of_train_val_set = len(train_input_img_paths) + len(val_input_img_paths)

for i in range(number_of_images_to_vis):
    img_index = i+len_of_train_val_set+1

    prefix = f"{img_index}."

    input_img_paths_i = find_images_with_prefix(test_input_img_paths_dir, prefix)
    target_img_paths_i = find_images_with_prefix(test_target_img_paths_dir, prefix)

    test_dataset_i = mod.get_dataset(
        batch_size, img_size, input_img_paths_i, target_img_paths_i
    )

    test_preds_i = model.predict(test_dataset_i)

    test_pred_list = []

    for i in range(len(test_preds_i)):
        generate_pred_list(i, test_preds_i)

    ground_truth_rect_i = cv2_visualizer.overlay_gd_and_rectangles(mask_dir_path="data/test-masks/",
                                                          file_path="data/test-visualized_masks/test_tiles.csv", image_number=img_index,
                                                          dir_path="data/test-images/",
                                                          save_path="data/test-visualized_masks/rectangles_",
                                                          mask_save_path="data/test-visualized_masks/merged_rectangles_with_mask_")

    pred_i = cv2_visualizer.merge_prediction_csv("data/test-visualized_masks/test_tiles.csv", test_pred_list, input_img_paths_i,
                                                  "data/test-images/", img_index, "data/test-visualized_masks/prediction_mask_",
                                                  "data/test-images/")

    final_mask_i = cv2_visualizer.final_mask_overlay("data/test-visualized_masks/prediction_mask_", img_index,
                                                      "data/test-visualized_masks/merged_rectangles_with_mask_",
                                                      "data/test-visualized_masks/final_mask_")




