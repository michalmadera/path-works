import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def load_path_from_file(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
        return data['positive']


def draw_square(ax, center, size):
    half_size = size / 2
    top_left = (int(center[0] - half_size), int(center[1] - half_size))
    bottom_right = (int(center[0] + half_size), int(center[1] + half_size))
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def draw_squares_on_path(image_path, mask_path, annotations, square_size, squares_data, tiles_folder,
                         mask_tiles_folder):
    img = Image.open(image_path)
    image = cv2.imread(image_path)
    img_mask = Image.open(mask_path)
    for annotation in annotations:
        vertices = np.array(annotation['vertices'], dtype=np.int32)
        cv2.polylines(image, [vertices], isClosed=True, color=(255, 0, 0), thickness=10)

        for i in range(len(vertices) - 1):
            start_point = vertices[i]
            end_point = vertices[i + 1]

            segment_length = distance(start_point, end_point)
            current_distance = 0

            while current_distance < segment_length:
                t = current_distance / segment_length if segment_length != 0 else 0
                inter_point = [start_point[0] + t * (end_point[0] - start_point[0]),
                               start_point[1] + t * (end_point[1] - start_point[1])]
                box_left = int(inter_point[0] - square_size/2)
                box_upper = int(inter_point[1] - square_size/2)
                box_right = int(inter_point[0] + square_size/2)
                box_lower = int(inter_point[1] + square_size/2)

                box = (box_left, box_upper, box_right, box_lower)

                tile = img.crop(box)
                mask_tile = img_mask.crop(box)
                save_tile(image_path, i, tile, tiles_folder)
                save_tile(mask_path, i, mask_tile, mask_tiles_folder, extension=".png")

                squares_data.append({
                    "slide_id": image_path,
                    "tile_index": i,
                    "tile_x": box_left,
                    "tile_y": box_upper,
                    "tile_width": square_size,
                    "tile_height": square_size
                })
                current_distance += square_size * 0.8

    return squares_data


def save_tile(image_path, index, tile, tiles_folder, extension=".jpg"):
    tile_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}.{index}{extension}"
    full_tile_filename = os.path.join(tiles_folder, tile_filename)
    tile.save(full_tile_filename)


def split_to_tiles(images_folder, tiles_folder, masks_folder, mask_tile_folder, csv_save_path, prefix="",
                   square_size=256, annotations_path="data/source-annotations"):
    os.makedirs(tiles_folder, exist_ok=True)
    os.makedirs(mask_tile_folder, exist_ok=True)
    df = pd.DataFrame()
    squares_data = []

    csv_save_path = os.path.join(prefix, csv_save_path)

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            input_path = os.path.join(images_folder, filename)
            mask_path = os.path.join(masks_folder, filename.replace(".jpg", ".png"))
            annotations = load_path_from_file(os.path.join(annotations_path, filename).replace(".jpg", ".json"))

            print(input_path, mask_path)

            squares_data = draw_squares_on_path(input_path, mask_path, annotations, square_size,
                                                squares_data, tiles_folder, mask_tile_folder)

    df = pd.DataFrame(squares_data)
    df.to_csv(csv_save_path, index=False)


if __name__ == "__main__":
    image = cv2.imread('../data/source-images/3.jpg')
    annotations = load_path_from_file('../data/source-annotations/3.json')
    squares_data = []
    draw_squares_on_path(image_path='../data/source-images/3.jpg', mask_path="../data/source-masks/3.png",
                         annotations=annotations, squares_data=squares_data, square_size=256,
                         tiles_folder="../data/outline-tiles", mask_tiles_folder="../data/outline-masks")
