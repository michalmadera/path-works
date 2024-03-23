import create_masks as masker
import image_tiles as tiler
import os

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

    prepare_masked_images()
    split_into_train_valid_test(300, 50, 50)
    split_to_tiles()

    print("Data prepared")

