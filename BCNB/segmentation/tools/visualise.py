from PIL import Image as PILImage, ImageOps, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import keras
import os
import re
import pandas as pd
import math

PILImage.MAX_IMAGE_PIXELS = None


def create_mask_list(input_img_list, predictions):
    output_mask_list = []
    for i in range(0, len(input_img_list)):
        mask = np.argmax(predictions[i], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = ImageOps.autocontrast(keras.utils.array_to_img(mask))
        output_mask_list.append(img)
    return output_mask_list


def merge_prediction_tiles(input_img_list, output_mask_list, image_path, tile_width, tile_height):
    tile_numbers = [os.path.splitext(os.path.basename(path))[0].split('.')[1] for path in input_img_list]
    tile_numbers = [int(tile_number) for tile_number in tile_numbers]

    image = PILImage.open(image_path)
    img_width, img_height = image.size

    tiles_per_column = img_width // tile_width
    tiles_per_row = img_height // tile_height

    predicted_mask = PILImage.new('RGB', (img_width, img_height), (0, 0, 0))
    for i in range(tiles_per_row * tiles_per_column):
        col = i // tiles_per_column
        row = i % tiles_per_column
        tile_index = i + 1
        if tile_index in tile_numbers:
            pred_tile = output_mask_list.pop(0)
            predicted_mask.paste(pred_tile, (col * tile_width, row * tile_height))
        else:
            black_tile = PILImage.new('RGB', (tile_width, tile_height), (0, 0, 0))  # Create black tile for missing ones
            predicted_mask.paste(black_tile, (col * tile_width, row * tile_height))
    return predicted_mask


def merge_prediction_csv(file_path, output_mask_list):
    df = pd.read_csv(file_path, sep=',')
    for index, row in df.iterrows():
        tile_x = row["tile_x"]
        tile_y = row["tile_y"]
        tile_width = row["tile_width"]
        tile_height = row["tile_height"]
        right = tile_x + tile_width
        lower = tile_y + tile_height
        box = (tile_x, tile_y, right, lower)


def overlay_masks(input_img_list, predictions, image_path, mask_path, tile_width,
                  tile_height):
    output_mask_list = create_mask_list(input_img_list, predictions)
    predicted_mask = merge_prediction_tiles(input_img_list, output_mask_list, image_path, tile_width, tile_height)

    image = PILImage.open(image_path)
    mask = PILImage.open(mask_path).convert("L")

    alpha = np.where(np.array(mask) == 0, 0, 255).astype(np.uint8)

    yellow_color = np.zeros_like(image)
    yellow_color[:, :, 0] = 64
    yellow_color[:, :, 1] = 64

    alpha_colored = np.where(alpha[..., None], yellow_color, 0)

    result = np.array(image) + alpha_colored

    merged_image = PILImage.fromarray(result.astype(np.uint8))

    mask_array = np.array(predicted_mask)
    mask_array[(mask_array == 255).all(axis=2)] = [0, 255, 0]

    green_mask = PILImage.fromarray(mask_array)
    green_mask_resized = green_mask.resize(image.size)

    merged_image_with_mask = PILImage.blend(merged_image.convert("RGBA"), green_mask_resized.convert("RGBA"), alpha=0)

    return merged_image_with_mask


def selected_image_path(image_number, dir_path, extension):
    return f"{dir_path}{image_number}.{extension}"


def merge_image_tiles_and_overlay(file_path, image_number, dir_path, save_path, save_mode, resize_size=(5000, 5000),
                                  resize=1):
    resized_save_path = selected_image_path(image_number, f"{save_path}resized_", "png")
    save_path = selected_image_path(image_number, save_path, "png")

    image = PILImage.open(selected_image_path(image_number, dir_path, "jpg"))
    img_rect = ImageDraw.Draw(image)
    df = pd.read_csv(file_path, sep=',')
    selected_rows = df.loc[df["slide_id"] == selected_image_path(image_number, dir_path, "jpg")]
    for index, row in selected_rows.iterrows():
        tile_x = row["tile_x"]
        tile_y = row["tile_y"]
        tile_width = row["tile_width"]
        tile_height = row["tile_height"]
        right = tile_x + tile_width
        lower = tile_y + tile_height
        box = (tile_x, tile_y, right, lower)
        img_rect.rectangle(box, outline="red", width=10)

    image_copy_for_resizing = image

    if save_mode == 1:
        save_merged_image(save_path, image)
        if resize == 1:
            resized_image = resize_image(image_copy_for_resizing, resize_size)
            save_merged_image(resized_save_path, resized_image)

    return image


def overlay_mask_with_rectangles(mask_dir_path, file_path, image_number, dir_path, save_path, mask_save_path,
                                 resize_size=(5000, 5000), resize=1, save_mode=0):
    mask_path = f"{mask_dir_path}{image_number}.png"
    image = merge_image_tiles_and_overlay(file_path, image_number, dir_path, save_path, save_mode)
    mask = PILImage.open(mask_path).convert("L")
    alpha = np.where(np.array(mask) == 0, 0, 255).astype(np.uint8)

    yellow_color = np.zeros_like(image)
    yellow_color[:, :, 0] = 64
    yellow_color[:, :, 1] = 64

    alpha_colored = np.where(alpha[..., None], yellow_color, 0)

    result = np.array(image) + alpha_colored

    merged_image = PILImage.fromarray(result.astype(np.uint8))
    save_path = selected_image_path(image_number, mask_save_path, "png")

    if resize == 1:
        merged_image = resize_image(merged_image, resize_size)

    merged_image.save(save_path)
    return merged_image

#funkcja do plotowania obrazków
def overlay_mask_with_rectangles_resize(mask_dir_path, file_path, image_number, dir_path, save_path, mask_save_path,
                                 resize_size=(500, 500), resize=1, save_mode=0):
    mask_path = f"{mask_dir_path}{image_number}.png"
    image = merge_image_tiles_and_overlay(file_path, image_number, dir_path, save_path, save_mode)
    image = resize_image(image, resize_size)
    mask = PILImage.open(mask_path).convert("L")
    mask = resize_image(mask, resize_size)
    alpha = np.where(np.array(mask) == 0, 0, 255).astype(np.uint8)

    yellow_color = np.zeros_like(image)
    yellow_color[:, :, 0] = 64
    yellow_color[:, :, 1] = 64

    alpha_colored = np.where(alpha[..., None], yellow_color, 0)

    result = np.array(image) + alpha_colored

    merged_image = PILImage.fromarray(result.astype(np.uint8))
    save_path = selected_image_path(image_number, mask_save_path, "png")

    return merged_image

def save_merged_image(save_path, merged_image):
    merged_image.save(save_path)


def display_merged_image(merged_image):
    plt.imshow(merged_image)
    plt.axis('off')
    plt.show()


def resize_image(image, size):
    image.thumbnail(size, PILImage.LANCZOS)
    return image


def plot_images(images):
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], aspect='auto')
        ax.axis('off')
        ax.text(0.5, -0.1, f"Slide {i}", transform=ax.transAxes, fontsize=10,
               ha='center', va='center')
    plt.show()


if __name__ == "__main__":
    input_img_paths = sorted([os.path.join("../data/test_tiles", fname) for fname in os.listdir("../data/test_tiles")])
    input_img_paths = sorted(input_img_paths,
                             key=lambda x: [int(i) if i.isdigit() else i for i in re.split('(\d+)', x)])
    predictions = []  #to jest lista predykcji modelu
    # images = []
    # directory = '../data/test_masks_tiles'
    # for filename in os.listdir(directory):
    #     if os.path.isfile(os.path.join(directory, filename)):
    #         image = PILImage.open(os.path.join(directory, filename))
    #         images.append(image)

    # merged_image = overlay_masks(image_path='../data/test-images/45.jpg', mask_path="../data/test-masks/45.png",
    #                              input_img_list=input_img_paths, tile_width=256, tile_height=256,
    #                              predictions=predictions)Wx`
    #merged_image_csv = merge_image_tiles_and_overlay( file_path="../data/visualized_masks/tiles.csv", image_number=19, dir_path="../data/source-images/", save_path="../data/visualized_masks/merged_image_with_rectangles_", save_mode=1)
    #save_merged_image("../data/visualized_masks/merged_image_with_mask.png", merged_image_csv)
    # merged_image_with_rectangels = overlay_mask_with_rectangles(mask_dir_path="../data/source-masks/",
    #                                                             file_path="../data/visualized_masks/tiles.csv",
    #                                                             image_number=19, dir_path="../data/source-images/",
    #                                                             save_path="../data/visualized_masks/merged_image_with_rectangles_",
    #                                                             mask_save_path="../data/visualized_masks/merged_rectangles_with_mask_")
    #display_merged_image(merged_image_csv)
    images = []
    for i in range(1, 7):
        image = overlay_mask_with_rectangles_resize(mask_dir_path="../data/source-masks/",
                                             file_path="../data/visualized_masks/tiles.csv",
                                             image_number=i, dir_path="../data/source-images/",
                                             save_path="../data/visualized_masks/merged_image_with_rectangles_",
                                             mask_save_path="../data/visualized_masks/merged_rectangles_with_mask_")
        images.append(image)

    resized_images = [resize_image(image, (50, 50)) for image in images]
    plot_images(resized_images)
