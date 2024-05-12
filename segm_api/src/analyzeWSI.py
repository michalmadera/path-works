import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
from fastapi import FastAPI, HTTPException, BackgroundTasks
import json
import asyncio
import segmentation_engine as engine
from pydantic import BaseModel
import time
import cv2
import numpy as np
import aiofiles
on_gpu = os.getenv('ON_GPU', 'false').lower() in ['true', '1', 't', 'y', 'yes']



app = FastAPI()

class AnalysisParameters(BaseModel):
    analysis_region_json: str
    is_normalized: bool


class AnalysisRequest(BaseModel):
    svs_path: str
    analysis_type: str
    analysis_parameters: AnalysisParameters

async def write_json(file_path, data):
    async with aiofiles.open(file_path, 'w') as file:
        await file.write(json.dumps(data))
        await file.flush()

async def read_json(file_path):
    async with aiofiles.open(file_path, 'r') as file:
        return json.loads(await file.read())

def calculate_bounding_box(coordinates):
    min_x = min(point['x'] for point in coordinates)
    max_x = max(point['x'] for point in coordinates)
    min_y = min(point['y'] for point in coordinates)
    max_y = max(point['y'] for point in coordinates)
    return (min_x, min_y, max_x, max_y)


async def create_mask_for_image(input_image_path, input_json_path, output_file_path, binary_mask=False):
    loop = asyncio.get_running_loop()
    if not os.path.exists(input_image_path):
        raise FileNotFoundError(f"No image found at {input_image_path}")

    def process_image():
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

    return await loop.run_in_executor(None, process_image)



async def cut_mask_from_array(image_array, json_path, min_x, min_y, save_path=None):
    loop = asyncio.get_running_loop()

    def process_array():
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

    return await loop.run_in_executor(None, process_array)


async def perform_analysis(svs_path: str, analysis_type: str, analysis_parameters: dict,
                           analysis_id: str):
    print(f"Czy aplikacja działa na GPU? {on_gpu}")

    analysis_dir = "analysis"
    results_dir = "/RESULTS"

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

    await write_json(json_file_path, json_data)

    try:
        start_time = time.time()
        # Użyj await dla asynchronicznych wywołań
        result, min_x, min_y = await create_mask_for_image(svs_path, analysis_parameters['analysis_region_json'],
                                                           mask_save_path)
        prediction = engine.make_prediction(svs_path=svs_path, location=[0, 0], on_gpu=on_gpu,
                                            size=[result.shape[1], result.shape[0]], save_path=save_path,
                                            save_dir=analysis_folder)
        triangle = await cut_mask_from_array(prediction, analysis_parameters['analysis_region_json'], min_x, min_y)
        end_time = time.time()
        prediction_time = end_time - start_time
        json_data["prediction_time"] = prediction_time

        start_time = time.time()
        engine.overlay_tif_with_pred(svs_path=svs_path, overlay=triangle, save_path=results_folder,
                                     location=[min_x, min_y])
        end_time = time.time()
        overlay_time = end_time - start_time
        json_data["overlay_time"] = overlay_time
        json_data["status"] = "finished"
        json_data["result_image_path"] = f"{results_folder}/result.tif"
    except Exception as e:
        json_data["status"] = "error"
        json_data["status_message"] = str(e)

    await write_json(json_file_path, json_data)


@app.post("/analyzeWSI")
async def analyze_wsi(background_tasks: BackgroundTasks, svs_path: str, analysis_type: str,
                      analysis_parameters_json: dict) -> str:
    analysis_parameters = AnalysisParameters(**analysis_parameters_json).dict()
    svs_path = svs_path.strip('"')

    results_dir = "/RESULTS"
    folders = os.listdir(results_dir)
    num_folders = len(folders)
    analysis_id = str(num_folders)

    background_tasks.add_task(perform_analysis, svs_path, analysis_type, analysis_parameters, analysis_id)

    return analysis_id

@app.get("/checkStatus")
async def check_status(analysis_id: str) -> str:
    file_path = find_results_path("/RESULTS", analysis_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File for analysis ID {analysis_id} not found")

    try:
        results_data = await read_json(file_path)
    except IOError:
        raise HTTPException(status_code=503, detail="Unable to read status file, try again later.")

    status = results_data['status']
    return status


@app.get("/resultsReady")
async def results_ready(analysis_id: str) -> dict:

    file_path = find_results_path("/RESULTS", analysis_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File for analysis ID {analysis_id} not found")

    with open(file_path, "r") as file:
        results_data = json.load(file)

    return results_data


def find_results_path(results_dir, analysis_id):

    results_dir = os.path.join(results_dir, f"{analysis_id}")

    file_path = os.path.join(results_dir, f"{analysis_id}.json")

    return file_path


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("analyzeWSI:app", host="0.0.0.0", port=8000, workers=4)
