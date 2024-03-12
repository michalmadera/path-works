import os
import numpy as np
from PIL import Image
import cv2
import pandas as pd

Image.MAX_IMAGE_PIXELS = None


def extract_tissue_mask(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.medianBlur(img, 31)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(blur)
    threshold, mask = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Threshold {threshold} for {image_path}")
    return mask


def split_image_into_tiles(image_path, tiles_folder, image_mask_path, mask_tiles_folder,
                           tile_width, tile_height, tissue_ratio_threshold, dataframe, first_path_segment):
    """
    Split an image into tiles and save them in a folder.

    :param tissue_ratio_threshold: The threshold to consider a tile as tissue.
    :param image_path: The path to the image to split
    :param tiles_folder: The folder to save the tiles.
    :param tile_width: The width of the tiles in pixels.
    :param tile_height: The height of the tiles in pixels.
    """
    # Load the main image
    img = Image.open(image_path)
    img_mask = Image.open(image_mask_path)
    img_width, img_height = img.size

    mask = extract_tissue_mask(image_path)
    disp_mask = np.copy(mask)

    # Calculate the number of tiles in each dimension
    x_tiles = img_width // tile_width
    y_tiles = img_height // tile_height

    # Split the image into tiles and save them
    index = 1
    for x in range(x_tiles):
        for y in range(y_tiles):
            # Define the box to crop
            left = x * tile_width
            upper = y * tile_height
            right = left + tile_width
            lower = upper + tile_height
            box = (left, upper, right, lower)

            # Crop the image to create the tile
            tile = img.crop(box)
            mask_tile = img_mask.crop(box)

            mask_area = mask[box[1]:box[3], box[0]:box[2]]
            content_area = (mask_area == 0).sum() / mask_area.size
            if content_area > tissue_ratio_threshold:
                disp_mask[box[1]:box[3], box[0]:box[2]] = disp_mask[box[1]:box[3], box[0]:box[2]] * 0.5
                # print(index, x, y, box, f"{content_area:.2f}")
                save_tile(image_path, index, tile, tiles_folder)
                save_tile(image_mask_path, index, mask_tile, mask_tiles_folder, extension=".png")
                dataframe = pd.concat([dataframe, pd.DataFrame([{"slide_id": os.path.join(first_path_segment,image_path), "tile_id": index, "tile_x": left, "tile_y": upper, "tile_height": tile_height, "tile_width": tile_width}])], ignore_index=True)
                # print(f"Tile saved to '{os.path.splitext(os.path.basename(image_path))[0]}.{index}.jpg, "
                #       f"ratio: {content_area:.2f}")

            index += 1

    return dataframe



    # cv2.imshow("window_name", disp_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def split_image_into_tiles_with_background(image_path, tiles_folder, image_mask_path, mask_tiles_folder,
                                           tile_width, tile_height):
    """
    Split an image into tiles and save them in a folder.

    :param mask_tiles_folder:
    :param image_mask_path:
    :param image_path: The path to the image to split
    :param tiles_folder: The folder to save the tiles.
    :param tile_width: The width of the tiles in pixels.
    :param tile_height: The height of the tiles in pixels.
    """
    # Load the main image
    img = Image.open(image_path)
    img_mask = Image.open(image_mask_path)
    img_width, img_height = img.size

    mask = extract_tissue_mask(image_path)
    disp_mask = np.copy(mask)

    # Calculate the number of tiles in each dimension
    x_tiles = img_width // tile_width
    y_tiles = img_height // tile_height

    # Split the image into tiles and save them
    index = 1
    for x in range(x_tiles):
        for y in range(y_tiles):
            # Define the box to crop
            left = x * tile_width
            upper = y * tile_height
            right = left + tile_width
            lower = upper + tile_height
            box = (left, upper, right, lower)

            # Crop the image to create the tile
            tile = img.crop(box)
            mask_tile = img_mask.crop(box)

            mask_area = mask[box[1]:box[3], box[0]:box[2]]

            save_tile(image_path, index, tile, tiles_folder)
            save_tile(image_mask_path, index, mask_tile, mask_tiles_folder, extension=".png")
            # print(f"Tile saved to '{os.path.splitext(os.path.basename(image_path))[0]}.{index}.jpg, "
            #       f"ratio: {content_area:.2f}")

            index += 1


def save_tile(image_path, index, tile, tiles_folder, extension=".jpg"):
    tile_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}.{index}{extension}"
    full_tile_filename = os.path.join(tiles_folder, tile_filename)
    tile.save(full_tile_filename)


def split_to_tiles(images_folder, tiles_folder, masks_folder, mask_tile_folder, tile_width, tile_height, csv_save_path, prefix,
                       tissue_ratio_threshold=.3):
        """
        Split all images in a folder into tiles and save them in another folder.

        :param csv_save_path: Path to save csv file
        :param images_folder: The folder containing the images to split.
        :param tiles_folder: The folder to save the tiles.
        :param tile_width: The width of the tiles in pixels.
        :param tile_height: The height of the tiles in pixels.
        """
        os.makedirs(tiles_folder, exist_ok=True)
        os.makedirs(mask_tile_folder, exist_ok=True)
        df = pd.DataFrame()
        for filename in os.listdir(images_folder):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                input_path = os.path.join(images_folder, filename)
                mask_path = os.path.join(masks_folder, filename.replace(".jpg", ".png"))
                print(input_path, mask_path)
                df = split_image_into_tiles(input_path, tiles_folder, mask_path, mask_tile_folder,
                                       tile_width, tile_height, tissue_ratio_threshold, df, prefix)
        df.to_csv(csv_save_path)

if __name__ == '__main__':
    split_image_into_tiles_with_background(("../wsi-segment/images/1.jpg", "../wsi-segment/test_tiles", "../wsi-segment/masks/1.jpg", "../wsi-segment/test_mask_tiles", 256, 256))
    split_to_tiles('../wsi-segment/images', '../wsi-segment/image-tiles',
                   '../wsi-segment/masks', '../wsi-segment/mask-tiles',
                   256, 256, csv_save_path='../data/visualized_masks/tile.csv', prefix="../")
    print("Tiles created and saved successfully.")
