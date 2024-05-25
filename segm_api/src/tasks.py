import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import json
import time
import cv2
import requests
import asyncio
import numpy as np
import pyvips
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

    on_gpu = os.getenv('ON_GPU', 'false').lower() in ['true', '1', 't', 'y', 'yes']
    print(f"Using GPU: {on_gpu}")
    api_url = os.getenv("results_ready_callback_url")
    print(f"Callback URL: {api_url}")

    analysis_dir = "analysis"
    results_dir = "/RESULTS"

    analysis_folder = os.path.join(analysis_dir, analysis_id)
    results_folder = os.path.join(results_dir, analysis_id)
    os.makedirs(analysis_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    mask_save_path = os.path.join(analysis_folder, "mask.tif")
    save_path = os.path.join(analysis_folder, "sample.tif")
    json_file_path = os.path.join(results_folder, f"{analysis_id}.json")

    # Load or create JSON file
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
        print(f"Creating mask for image: {svs_path}")
        result, min_x, min_y = create_mask_for_image(svs_path, analysis_parameters['analysis_region_json'], mask_save_path)
        print(f"Creating prediction")
        prediction = engine.make_prediction(svs_path=svs_path, location=[min_x, min_y], on_gpu=on_gpu, size=[result.shape[1], result.shape[0]], save_path=save_path, save_dir=analysis_folder)
        print(f"Cutting mask from prediction")
        triangle = cut_mask_from_array(prediction, analysis_parameters['analysis_region_json'], min_x, min_y)
        end_time = time.time()
        prediction_time = end_time - start_time
        json_data["prediction_time"] = prediction_time

        start_time = time.time()
        print(f"Overlaying prediction on image")
        engine.overlay_tif_with_pred(svs_path=svs_path, overlay=triangle, save_path=results_folder, location=[min_x, min_y])
        end_time = time.time()
        overlay_time = end_time - start_time
        json_data["overlay_time"] = overlay_time
        json_data["status"] = "finished"
        json_data["result_image_path"] = f"{results_folder}/result.tif"
    except Exception as e:
        json_data["status"] = "error"
        json_data["status_message"] = str(e)
    finally:
        with open(json_file_path, 'w') as file:
            json.dump(json_data, file)
        # Call resultsReady service
        try:
            print(f"Attempting to call results ready for analysis_id: {analysis_id}")
            asyncio.run(call_results_ready(analysis_id, api_url))
        except Exception as e:
            print(f"Error while calling call_results_ready: {str(e)}")
    print(f"Analysis completed: {analysis_id}")

def create_mask_for_image(input_image_path, input_json_path, output_file_path, binary_mask=False):
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"No image found at {input_image_path}")

    reader = WSIReader.open(input_image_path, power=1)
    original_image = reader.slide_thumbnail(resolution=1, units="power")
    if original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)

    with open(input_json_path, 'r') as file:
        data = json.load(file)

    coordinates = data['features'][0]['geometry']['coordinates']
    min_x, min_y, max_x, max_y = calculate_bounding_box(coordinates)
    cropped_image = original_image[min_y:max_y, min_x:max_x]

    # Ensure the array is C-contiguous
    cropped_image = np.ascontiguousarray(cropped_image)

    # Save as TIFF
    cropped_image_vips = pyvips.Image.new_from_memory(cropped_image.data, cropped_image.shape[1], cropped_image.shape[0], cropped_image.shape[2], 'uchar')
    cropped_image_vips.write_to_file(output_file_path.replace('.jpg', '.tif'), tile=True, compression="jpeg")

    return cropped_image, min_x, min_y

def cut_mask_from_array(image_array, json_path, min_x, min_y, save_path=None):
    with open(json_path, 'r') as file:
        data = json.load(file)

    coordinates = data['features'][0]['geometry']['coordinates']
    if isinstance(coordinates[0][0], list):
        coordinates = coordinates[0]
    adjusted_triangle = [(int(coord[0]) - min_x, int(coord[1]) - min_y) for coord in coordinates]

    mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(adjusted_triangle)], 255)

    # Convert mask to 3 channels if necessary
    if image_array.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    result = cv2.bitwise_and(image_array, image_array, mask=mask)
    if save_path:
        # Ensure the array is C-contiguous
        result = np.ascontiguousarray(result)

        # Save as TIFF
        result_vips = pyvips.Image.new_from_memory(result.data, result.shape[1], result.shape[0], result.shape[2] if result.ndim == 3 else 1, 'uchar')
        result_vips.write_to_file(save_path.replace('.jpg', '.tif'), tile=True, compression="jpeg")

    return result