import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import json
import time
import cv2
import requests
import asyncio
import numpy as np
from tiatoolbox.wsicore.wsireader import WSIReader
from .celery_app import celery_app
from . import segmentation_engine as engine
from pydantic import BaseModel


class AnalysisParameters(BaseModel):
    analysis_region_json: str
    is_normalized: bool


def calculate_bounding_box(coordinates):
    if isinstance(coordinates[0][0], list):
        coordinates = coordinates[0]
    min_x = min(int(coord[0]) for coord in coordinates)
    max_x = max(int(coord[0]) for coord in coordinates)
    min_y = min(int(coord[1]) for coord in coordinates)
    max_y = max(int(coord[1]) for coord in coordinates)
    return (min_x, min_y, max_x, max_y)


async def call_results_ready(analysis_id: str, api_url):
    payload = {
        "analysis_id": analysis_id
    }
    print(f"Calling results ready callback for analysis_id: {analysis_id} to URL: {api_url}")
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            print(f"Successfully called results ready callback for analysis_id: {analysis_id}")
        else:
            print(f"Error calling results ready callback: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Exception during calling results ready callback: {str(e)}")


@celery_app.task(bind=True, name='tasks.perform_analysis')
def perform_analysis(self, svs_path: str, analysis_type: int, analysis_parameters: dict):
    analysis_id = self.request.id
    print(f"Started analysis: {analysis_id}")

    api_url = os.getenv("results_ready_callback_url")
    print(f"Callback URL: {api_url}")

    analysis_dir = "analysis"
    results_dir = "/RESULTS"

    analysis_folder = os.path.join(analysis_dir, analysis_id)
    results_folder = os.path.join(results_dir, analysis_id)
    os.makedirs(analysis_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    json_file_path = os.path.join(results_folder, f"{analysis_id}.json")

    try:
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
    except IOError:
        json_data = {
            "analysis_id": analysis_id,
            "svs_path": svs_path,
            "analysis_type": analysis_type,
            "region_json_path": analysis_parameters['analysis_region_json'],
            "is_normalized": analysis_parameters['is_normalized'],
            "status": "in_process",
            "result_json_path": json_file_path,
        }

    with open(json_file_path, 'w') as file:
        json_data["status"] = "in_process"
        json.dump(json_data, file)

    try:
        start_time = time.time()

        print(f"Reading region from image: {svs_path}")

        with open(analysis_parameters['analysis_region_json'], 'r') as file:
            region_data = json.load(file)

        coordinates = region_data['features'][0]['geometry']['coordinates']
        min_x, min_y, max_x, max_y = calculate_bounding_box(coordinates)
        region_width = max_x - min_x
        region_height = max_y - min_y

        print(f"Region size: {region_width}x{region_height} at position [{min_x}, {min_y}]")

        print(f"Making prediction")

        masks = engine.make_prediction(
            svs_path=svs_path,
            location=[min_x, min_y],
            size=[region_width, region_height],
            save_dir=analysis_folder
        )

        end_time = time.time()
        prediction_time = end_time - start_time
        json_data["prediction_time"] = prediction_time

        mask_paths = {}
        for class_name, mask in masks.items():
            if isinstance(coordinates[0][0], list):
                adjusted_coordinates = [(int(coord[0]) - min_x, int(coord[1]) - min_y) for coord in coordinates[0]]
            else:
                adjusted_coordinates = [(int(coord[0]) - min_x, int(coord[1]) - min_y) for coord in coordinates]

            poly_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillPoly(poly_mask, [np.array(adjusted_coordinates)], 255)

            masked_result = cv2.bitwise_and(mask, poly_mask)

            mask_path = os.path.join(results_folder, f"{class_name.lower()}_mask.npy")
            np.save(mask_path, masked_result)
            mask_paths[class_name] = mask_path

        json_data["mask_paths"] = mask_paths
        json_data["status"] = "finished"

    except Exception as e:
        json_data["status"] = "error"
        json_data["status_message"] = str(e)
    finally:
        with open(json_file_path, 'w') as file:
            json.dump(json_data, file)
        try:
            print(f"Attempting to call results ready for analysis_id: {analysis_id}")
            asyncio.run(call_results_ready(analysis_id, api_url))
        except Exception as e:
            print(f"Error while calling call_results_ready: {str(e)}")
    print(f"Analysis completed: {analysis_id}")


def extract_region(input_image_path, input_json_path, binary_mask=False):
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"No image found at {input_image_path}")

    reader = WSIReader.open(input_image_path, power=1)

    with open(input_json_path, 'r') as file:
        data = json.load(file)

    coordinates = data['features'][0]['geometry']['coordinates']
    min_x, min_y, max_x, max_y = calculate_bounding_box(coordinates)

    region_width = max_x - min_x
    region_height = max_y - min_y
    cropped_image = reader.read_region(
        location=[min_x, min_y],
        level=0,
        size=[region_width, region_height]
    )

    if cropped_image.shape[2] == 4:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2BGR)

    return cropped_image, min_x, min_y