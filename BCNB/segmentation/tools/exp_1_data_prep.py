import shutil

import create_masks as masker
import image_tiles as tiler
import os
import re

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in filter(None, re.split('([0-9]+)', key))]
    return sorted(data, key=alphanum_key)

def prepare_masked_images():
    masker.create_masks_for_folder("data/source-images",
                                   "data/source-annotations",
                                   "data/source-masks")

def split_into_train_valid_test(no_train_images, no_val_images, no_test_images):
    os.makedirs("data/train-images", exist_ok=True)
    os.makedirs("data/val-images", exist_ok=True)
    os.makedirs("data/test-images", exist_ok=True)
# TODO: Implement split_into_train_valid_test.
# Take first no_train_images  from data/source-images and move them to data/train-images
# Take next no_val_images from data/source-images and move them to data/val-images
# Take next no_test_images from data/source-images and move them to data/test-images
    source_dir = "data/source-images"
    train_dir = "data/train-images"
    val_dir = "data/val-images"
    test_dir = "data/test-images"
    files = sorted_alphanumeric(os.listdir(source_dir))

    val_file_range = no_train_images + no_val_images
    test_file_range = val_file_range + no_test_images

    #przenoszenie pierwszych N plików do folderu train-images
    for filename in files[:no_train_images]:
        shutil.move(os.path.join(source_dir, filename), os.path.join(train_dir, filename))

    # przenoszenie kolejnych M plików do folderu val-images
    for filename in files[no_train_images:val_file_range]:
        shutil.move(os.path.join(source_dir, filename), os.path.join(val_dir, filename))

    # przenoszenie kolejnych X plików do folderu test-images
    for filename in files[val_file_range:test_file_range]:
        shutil.move(os.path.join(source_dir, filename), os.path.join(test_dir, filename))

def split_to_tiles():
    tiler.split_to_tiles("data/train-images",
                         "data/train-image-tiles",
                         "data/train-masks",
                         "data/train-mask-tiles",
                         256, 256,
                         "data/train-visualized_masks", "")

    tiler.split_to_tiles("data/valid-images",
                         "data/valid-image-tiles",
                         "data/valid-masks",
                         "data/valid-mask-tiles",
                         256, 256,
                         "data/valid-visualized_masks", "")

    tiler.split_to_tiles("data/test-images",
                         "data/test-image-tiles",
                         "data/test-masks",
                         "data/test-mask-tiles",
                         256, 256,
                         "data/test-visualized_masks", "")



if __name__ == '__main__':

    #prepare_masked_images()
    split_into_train_valid_test(20, 5, 5)
    #split_to_tiles()


