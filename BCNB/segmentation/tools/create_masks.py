import json
import numpy as np
import cv2
import os

# Path to your JSON file and image
# json_file_path = '../wsi-segment/source-images-annotated/1.json'
# image_file_path = '../wsi-segment/source-images/1.jpg'
# new_image_path = '../wsi-segment/source-masks/1.jpg'


def create_mask_for_image(input_image_path, input_json_path, output_file_path, binary_mask=False):

    with open(input_json_path, 'r') as file:
        data = json.load(file)

    # Load the original image to get its size
    print('input_image_path: ', input_image_path)
    original_image = cv2.imread(input_image_path)
    height, width = original_image.shape[:2]
    mask_image = np.ones((height, width), np.uint8) * 0

    int_vertices = []
    # Process 'positive' images-annotated and draw white polygons
    for annotation in data['positive']:
        vertices = annotation['vertices']
        int_vertices = [(int(x), int(y)) for [x, y] in vertices]
        if len(int_vertices) > 2:  # Need at least 3 points to draw a polygon
            cv2.fillPoly(mask_image, [np.array(int_vertices)], color=255)

    # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    # mask_image[mask_image == 255] = 1
    cv2.imwrite(output_file_path, mask_image)


def create_masks_for_folder(source_image_folder, source_json_folder, output_folder, binary_mask=False):
    os.makedirs(output_folder, exist_ok=True)

    # TODO: Add a check for the number of files in the source_image_folder and source_json_folder.
    #  A difference would be a problem.

    for filename in os.listdir(source_json_folder):
        if filename.endswith(".json"):
            input_json_path = os.path.join(source_json_folder, filename)
            mask_filename = f"{os.path.splitext(os.path.basename(input_json_path))[0]}.jpg"
            output_file_path = os.path.join(output_folder, mask_filename)
            input_image_path = os.path.join(source_image_folder, mask_filename)
            # print(input_image_path, input_json_path, output_file_path)
            create_mask_for_image(input_image_path, input_json_path, output_file_path, binary_mask)


if __name__ == '__main__':
    source_json_folder = '../data/source-annotations'
    source_image_folder = '../data/source-images'
    output_folder = '../data/source-masks'
    create_masks_for_folder(source_image_folder, source_json_folder, output_folder)
    # input_image_path = "../data/source-images/1.jpg"
    # input_json_path = "../data/source-annotations/1.json"
    # output_file_path = "../data/source-masks/1.jpg"
    # create_mask_for_image(input_json_path=input_json_path, input_image_path=input_image_path, output_file_path=output_file_path)
    # print('Done')

# m = cv2.imread("../wsi-segment/mask-tiles/1.1083.jpg")
# m.shape
# m[1]


