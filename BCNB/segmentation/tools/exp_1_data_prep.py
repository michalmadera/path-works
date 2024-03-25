import shutil

import create_masks as masker
import image_tiles as tiler
import os
import re
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in filter(None, re.split('([0-9]+)', key))]
    return sorted(data, key=alphanum_key)

def prepare_masked_images():
    images_list = os.listdir("data/source-images")
    mask_list = os.listdir("data/source-masks")
    if len(mask_list) != len(images_list):
        masker.create_masks_for_folder("data/source-images",
                                       "data/source-annotations",
                                       "data/source-masks")

def split_into_train_valid_test(no_train_images, no_val_images, no_test_images):
    os.makedirs("data/train-images", exist_ok=True)
    os.makedirs("data/valid-images", exist_ok=True)
    os.makedirs("data/test-images", exist_ok=True)
    os.makedirs("data/train-masks", exist_ok=True)
    os.makedirs("data/valid-masks", exist_ok=True)
    os.makedirs("data/test-masks", exist_ok=True)
    os.makedirs("data/train-visualized_masks", exist_ok=True)
    os.makedirs("data/test-visualized_masks", exist_ok=True)
    os.makedirs("data/valid-visualized_masks", exist_ok=True)
# TODO: Implement split_into_train_valid_test.
# Take first no_train_images  from data/source-images and move them to data/train-images
# Take next no_val_images from data/source-images and move them to data/val-images
# Take next no_test_images from data/source-images and move them to data/test-images
    source_dir = "data/source-images"
    mask_source_dir = "data/source-masks"
    train_dir = "data/train-images"
    val_dir = "data/valid-images"
    test_dir = "data/test-images"
    mask_train_dir = "data/train-masks"
    mask_val_dir = "data/valid-masks"
    mask_test_dir = "data/test-masks"

    files = sorted_alphanumeric(os.listdir(source_dir))
    masks = sorted_alphanumeric(os.listdir(mask_source_dir))

    val_file_range = no_train_images + no_val_images
    test_file_range = val_file_range + no_test_images

    #przenoszenie pierwszych N plików do folderu train-images
    for filename in files[:no_train_images]:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(train_dir, filename))

    # przenoszenie kolejnych M plików do folderu val-images
    for filename in files[no_train_images:val_file_range]:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(val_dir, filename))

    # przenoszenie kolejnych X plików do folderu test-images
    for filename in files[val_file_range:test_file_range]:
        shutil.copy(os.path.join(source_dir, filename), os.path.join(test_dir, filename))

    #MASKI
    for filename in masks[:no_train_images]:
        shutil.copy(os.path.join(mask_source_dir, filename), os.path.join(mask_train_dir, filename))

    for filename in masks[no_train_images:val_file_range]:
        shutil.copy(os.path.join(mask_source_dir, filename), os.path.join(mask_val_dir, filename))

    for filename in masks[val_file_range:test_file_range]:
        shutil.copy(os.path.join(mask_source_dir, filename), os.path.join(mask_test_dir, filename))

def split_to_tiles():
    tiler.split_to_tiles("data/train-images",
                         "data/train-image-tiles",
                         "data/train-masks",
                         "data/train-mask-tiles",
                         256, 256,
                         "data/train-visualized_masks/train_tiles.csv", "")

    tiler.split_to_tiles("data/valid-images",
                         "data/valid-image-tiles",
                         "data/valid-masks",
                         "data/valid-mask-tiles",
                         256, 256,
                         "data/valid-visualized_masks/valid_tiles.csv", "")

    tiler.split_to_tiles("data/test-images",
                         "data/test-image-tiles",
                         "data/test-masks",
                         "data/test-mask-tiles",
                         256, 256,
                         "data/test-visualized_masks/test_tiles.csv", "")



if __name__ == '__main__':

    prepare_masked_images()
    split_into_train_valid_test(100, 10, 10)
    split_to_tiles()


