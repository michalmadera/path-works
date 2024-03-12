from cv2 import rectangle, imread, imwrite
import cv2
import pandas as pd
import numpy as np
import os

def selected_image_path(image_number, dir_path, extension):
    return f"{dir_path}{image_number}.{extension}"
def draw_rectangles(file_path: str, image_number: int, dir_path: str, save_path: str, save_mode: int,
                    resize_size: tuple = (5000, 5000), resize: int = 1) -> None:
    resized_save_path = selected_image_path(image_number, f"{save_path}resized_", "png")
    save_path = selected_image_path(image_number, save_path, "png")

    image = imread(selected_image_path(image_number, dir_path, "jpg"))
    df = pd.read_csv(file_path, sep=',')
    selected_rows = df.loc[df["slide_id"] == selected_image_path(image_number, dir_path, "jpg")]
    for index, row in selected_rows.iterrows():
        tile_x = row["tile_x"]
        tile_y = row["tile_y"]
        tile_width = row["tile_width"]
        tile_height = row["tile_height"]
        right = tile_x + tile_width
        lower = tile_y + tile_height
        rectangle(image, (tile_x, tile_y), (right, lower), (0, 0, 255), 10)

    image_copy_for_resizing = image.copy()

    if save_mode == 1:
        imwrite(save_path, image)
        if resize == 1:
            resized_image = resize_image(image_copy_for_resizing, resize_size)
            imwrite(resized_save_path, resized_image)

    return image


def overlay_gd_and_rectangles(mask_dir_path: str, file_path: str, image_number: int, dir_path: str,
                              save_path: str, mask_save_path: str, resize_size: tuple = (5000, 5000),
                              resize: int = 0, save_mode: int = 1):
    mask_path = f"{mask_dir_path}{image_number}.png"
    image = draw_rectangles(file_path, image_number, dir_path, save_path, save_mode)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, alpha = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    alpha = cv2.merge([alpha, alpha, alpha])

    yellow_color = np.zeros_like(image, dtype=np.uint8)
    yellow_color[:, :, 0] = 64
    yellow_color[:, :, 2] = 64

    alpha_colored = cv2.bitwise_and(yellow_color, alpha)

    result = cv2.add(image, alpha_colored)

    save_path = selected_image_path(image_number, mask_save_path, "png")

    if resize == 1:
        result = cv2.resize_image(result, resize_size)

    cv2.imwrite(save_path, result)
    return result

def merge_prediction_csv(file_path: str, output_mask_list: list, input_img_list: list, image_path: str,
                         image_number: int, save_path: str, dir_path: str):
    df = pd.read_csv(file_path, sep=',')
    save_path = selected_image_path(image_number, save_path, "png")

    tile_numbers = [int(os.path.splitext(os.path.basename(path))[0].split('.')[1]) for path in input_img_list]
    image_path = selected_image_path(image_number, image_path, "jpg")
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    predicted_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    selected_rows = df.loc[df["slide_id"] == selected_image_path(image_number, dir_path, "jpg")]
    for index, row in selected_rows.iterrows():
        tile_x = row["tile_x"]
        tile_y = row["tile_y"]
        tile_width = row["tile_width"]
        tile_height = row["tile_height"]
        tile_id = row["tile_id"]
        right = tile_x + tile_width
        lower = tile_y + tile_height
        box = (tile_x, tile_y, right, lower)
        if tile_id in tile_numbers:
            pred_tile = output_mask_list.pop(0)
            pred_tile = np.array(pred_tile)
            pred_tile_resized = cv2.resize(pred_tile, (tile_width, tile_height))
            pred_tile_resized = cv2.cvtColor(pred_tile_resized, cv2.COLOR_GRAY2RGB)  # Konwersja do obrazu RGB
            predicted_mask[tile_y:lower, tile_x:right] = pred_tile_resized

    cv2.imwrite(save_path, predicted_mask)
    return predicted_mask

def final_mask_overlay(mask_dir_path, image_number, image_dir, save_path, resize_size=(5000, 5000), resize=1):
    mask_path = f"{mask_dir_path}{image_number}.png"
    img_path = f"{image_dir}{image_number}.png"

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    result = img.copy()

    alpha = np.where(mask == 0, 0, 255).astype(np.uint8)

    green_color = np.zeros_like(img)
    green_color[:, :, 1] = 255

    alpha_colored = cv2.bitwise_and(green_color, green_color, mask=alpha)

    result = cv2.add(result, alpha_colored)

    save_path = selected_image_path(image_number, save_path, "png")
    if resize == 1:
        result = resize_image(result, resize_size)
    cv2.imwrite(save_path, result)

    return result


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    h, w = image.shape[:2]
    new_w, new_h = size

    # Obliczanie współczynnika proporcji
    if w > h:
        new_h = int(h * new_w / w)
    else:
        new_w = int(w * new_h / h)

    # Skalowanie obrazu
    resized_image = cv2.resize(image, (new_w, new_h))

    return resized_image