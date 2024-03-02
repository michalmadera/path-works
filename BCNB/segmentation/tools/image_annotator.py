import json
import numpy as np
import cv2
import os

# Path to your JSON file and image
# json_file_path = '../wsi-segment/source-images-annotated/1.json'
# image_file_path = '../wsi-segment/source-images/1.jpg'
# new_image_path = '../wsi-segment/source-masks/1.jpg'


def add_annotation_to_image(input_image_path, input_json_path, output_file_path):

    with open(input_json_path, 'r') as file:
        data = json.load(file)

    # Load the original image to get its size
    print('input_image_path: ', input_image_path)
    original_image = cv2.imread(input_image_path)
    height, width = original_image.shape[:2]

    # Create a new black image of the same size
    annotated_image = np.zeros((height, width, 3), np.uint8)

    # Process 'positive' images-annotated and draw white polygons
    for annotation in data['positive']:
        vertices = annotation['vertices']
        int_vertices = [(int(x), int(y)) for [x, y] in vertices]
        if len(int_vertices) > 2:  # Need at least 3 points to draw a polygon
            annotated_image = cv2.polylines(original_image, [np.array(int_vertices)], True, (255, 0, 0), 20)

    cv2.imwrite(output_file_path, annotated_image)


def create_image_annotations_for_folder(source_image_folder, source_json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # TODO: Add a check for the number of files in the source_image_folder and source_json_folder.
    #  A difference would be a problem.

    for filename in os.listdir(source_json_folder):
        if filename.endswith(".json"):
            input_json_path = os.path.join(source_json_folder, filename)
            mask_filename = f"{os.path.splitext(os.path.basename(input_json_path))[0]}.jpg"
            output_file_path = os.path.join(output_folder, mask_filename)
            input_image_path = output_file_path.replace(output_folder, source_image_folder)
            print(input_image_path, input_json_path, output_file_path)
            add_annotation_to_image(input_image_path, input_json_path, output_file_path)


if __name__ == '__main__':
    source_json_folder = '../wsi-segment/source-annotations'
    source_image_folder = '../wsi-segment/source-images'
    output_folder = '../wsi-segment/source-images-annotated'
    create_image_annotations_for_folder(source_image_folder, source_json_folder, output_folder)
