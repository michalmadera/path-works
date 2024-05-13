import os
import json
import time
import cv2
import numpy as np
from celery_app import celery_app
import segmentation_engine as engine
from pydantic import BaseModel

class AnalysisParameters(BaseModel):
    analysis_region_json: str
    is_normalized: bool

def calculate_bounding_box(coordinates):
    min_x = min(point['x'] for point in coordinates)
    max_x = max(point['x'] for point in coordinates)
    min_y = min(point['y'] for point in coordinates)
    max_y = max(point['y'] for point in coordinates)
    return (min_x, min_y, max_x, max_y)

@celery_app.task(bind=True, name='tasks.perform_analysis')
def perform_analysis(self, svs_path: str, analysis_type: int, analysis_parameters: dict, analysis_id: str):
    print(f"Rozpoczęto analizę: {analysis_id}")

    on_gpu = os.getenv('ON_GPU', 'false').lower() in ['true', '1', 't', 'y', 'yes']

    analysis_dir = "analysis"
    results_dir = "RESULTS"

    analysis_folder = os.path.join(analysis_dir, analysis_id)
    results_folder = os.path.join(results_dir, analysis_id)
    os.makedirs(analysis_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    mask_save_path = os.path.join(analysis_folder, "mask.jpg")
    save_path = os.path.join(analysis_folder, "sample.jpg")
    json_file_path = os.path.join(results_folder, f"{analysis_id}.json")

    json_data = {
        "analysis_id": analysis_id,
        "svs_path": svs_path,
        "analysis_type": analysis_type,
        "region_json_path": analysis_parameters['analysis_region_json'],
        "is_normalized": analysis_parameters['is_normalized'],
        "status": "in_process",
        "result_json_path": json_file_path,
        "on_gpu": on_gpu
    }

    with open(json_file_path, 'w') as file:
        json.dump(json_data, file)

    try:
        start_time = time.time()
        print(f"Tworzenie maski dla obrazu: {svs_path}")
        result, min_x, min_y = create_mask_for_image(svs_path, analysis_parameters['analysis_region_json'], mask_save_path)
        print(f"Tworzenie predykcji")
        prediction = engine.make_prediction(svs_path=svs_path, location=[0, 0], on_gpu=on_gpu, size=[result.shape[1], result.shape[0]], save_path=save_path, save_dir=analysis_folder)
        print(f"Przycinanie maski z predykcji")
        triangle = cut_mask_from_array(prediction, analysis_parameters['analysis_region_json'], min_x, min_y)
        end_time = time.time()
        prediction_time = end_time - start_time
        json_data["prediction_time"] = prediction_time

        start_time = time.time()
        print(f"Nakładanie predykcji na obraz")
        engine.overlay_tif_with_pred(svs_path=svs_path, overlay=triangle, save_path=results_folder, location=[min_x, min_y])
        end_time = time.time()
        overlay_time = end_time - start_time
        json_data["overlay_time"] = overlay_time
        json_data["status"] = "finished"
        json_data["result_image_path"] = f"{results_folder}/result.tif"
    except Exception as e:
        json_data["status"] = "error"
        json_data["status_message"] = str(e)

    with open(json_file_path, 'w') as file:
        json.dump(json_data, file)
    print(f"Zakończono analizę: {analysis_id}")

def create_mask_for_image(input_image_path, input_json_path, output_file_path, binary_mask=False):
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"No image found at {input_image_path}")

    original_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if original_image.shape[2] == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2BGRA)

    with open(input_json_path, 'r') as file:
        data = json.load(file)

    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    coordinates = data['geometry']['coordinates']
    min_x, min_y, max_x, max_y = calculate_bounding_box(coordinates)
    cropped_image = original_image[min_y:max_y, min_x:max_x]
    cv2.imwrite(output_file_path, cropped_image)
    return cropped_image, min_x, min_y

def cut_mask_from_array(image_array, json_path, min_x, min_y, save_path=None):
    with open(json_path, 'r') as file:
        data = json.load(file)

    coordinates = data['geometry']['coordinates']
    adjusted_triangle = [(int(coord['x']) - min_x, int(coord['y']) - min_y) for coord in coordinates]

    mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(adjusted_triangle)], 255)

    result = cv2.bitwise_and(image_array, image_array, mask=mask)
    if save_path:
        cv2.imwrite(save_path, result)

    return result
