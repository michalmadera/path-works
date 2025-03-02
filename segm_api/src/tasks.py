import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import json
import time
import cv2
import aiohttp
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from pydantic import BaseModel

from tiatoolbox import logger
from tiatoolbox.wsicore.wsireader import WSIReader
from .celery_app import celery_app
from . import segmentation_engine as engine


class AnalysisParameters(BaseModel):
    analysis_region_json: str
    is_normalized: bool


def calculate_bounding_box(coordinates: List[List[float]]) -> Tuple[int, int, int, int]:
    """Oblicza prostokąt ograniczający dla zestawu współrzędnych."""
    if isinstance(coordinates[0][0], list):
        coordinates = coordinates[0]
    coords = np.array(coordinates, dtype=np.int32)
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    return min_x, min_y, max_x, max_y


async def call_results_ready(analysis_id: str, api_url: str) -> None:
    """Asynchroniczne wywołanie zwrotne z informacją o gotowości wyników."""
    payload = {"analysis_id": analysis_id}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"Successfully called results ready callback for analysis_id: {analysis_id}")
                else:
                    logger.error(f"Error calling results ready callback: {response.status} - {await response.text()}")
    except Exception as e:
        logger.error(f"Exception during calling results ready callback: {str(e)}")


def create_mask_for_image(
        input_image_path: str,
        input_json_path: str,
        output_file_path: str,
        binary_mask: bool = False
) -> Tuple[np.ndarray, int, int]:
    """Tworzy maskę dla obrazu na podstawie pliku JSON z regionem."""
    if not Path(input_image_path).exists():
        raise FileNotFoundError(f"No image found at {input_image_path}")

    wsi_reader = WSIReader.open(input_img=input_image_path)
    original_image = wsi_reader.slide_thumbnail(resolution=1, units="power")

    with open(input_json_path, 'r') as file:
        data = json.load(file)

    coordinates = data['features'][0]['geometry']['coordinates']
    min_x, min_y, max_x, max_y = calculate_bounding_box(coordinates)

    cropped_image = original_image[min_y:max_y, min_x:max_x]

    if cropped_image.shape[2] == 3:
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)

    # Zapisz jako PNG
    output_file_path = str(Path(output_file_path).with_suffix('.png'))
    cv2.imwrite(output_file_path, cropped_image)

    return cropped_image, min_x, min_y


def cut_mask_from_array(
        image_array: np.ndarray,
        json_path: str,
        min_x: int,
        min_y: int,
        save_path: str = None
) -> np.ndarray:
    """Wycina maskę z tablicy obrazu na podstawie współrzędnych z pliku JSON."""
    with open(json_path, 'r') as file:
        data = json.load(file)

    coordinates = data['features'][0]['geometry']['coordinates']
    if isinstance(coordinates[0][0], list):
        coordinates = coordinates[0]

    adjusted_triangle = np.array([
        (int(coord[0]) - min_x, int(coord[1]) - min_y)
        for coord in coordinates
    ], dtype=np.int32)

    mask = np.zeros(image_array.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [adjusted_triangle], 255)

    result = cv2.bitwise_and(image_array, image_array, mask=mask)

    if save_path:
        save_path = str(Path(save_path).with_suffix('.png'))
        cv2.imwrite(save_path, result)

    return result


@celery_app.task(bind=True, name='tasks.perform_analysis')
def perform_analysis(self, svs_path: str, analysis_type: int, analysis_parameters: dict):
    """Główne zadanie Celery wykonujące analizę obrazu."""
    analysis_id = self.request.id
    logger.info(f"Started analysis: {analysis_id}")

    # Konfiguracja
    on_gpu = os.getenv('ON_GPU', 'false').lower() in ['true', '1', 't', 'y', 'yes']
    api_url = os.getenv("results_ready_callback_url")
    logger.info(f"Using GPU: {on_gpu}")
    logger.info(f"Callback URL: {api_url}")

    # Przygotowanie katalogów
    analysis_dir = Path("analysis")
    results_dir = Path("/RESULTS")

    analysis_folder = analysis_dir / analysis_id
    results_folder = results_dir / analysis_id

    analysis_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    # Ścieżki plików
    mask_save_path = analysis_folder / "mask.png"
    save_path = analysis_folder / "sample.png"
    json_file_path = results_folder / f"{analysis_id}.json"

    # Inicjalizacja lub wczytanie pliku JSON
    try:
        with open(json_file_path, 'r') as file:
            json_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        json_data = {
            "analysis_id": analysis_id,
            "svs_path": svs_path,
            "analysis_type": analysis_type,
            "region_json_path": analysis_parameters['analysis_region_json'],
            "is_normalized": analysis_parameters['is_normalized'],
            "status": "in_process",
            "result_json_path": str(json_file_path),
        }

    try:
        # Zapisz status początkowy
        json_data["status"] = "in_process"
        with open(json_file_path, 'w') as file:
            json.dump(json_data, file)

        start_time = time.time()

        # Tworzenie maski dla obrazu
        logger.info(f"Creating mask for image: {svs_path}")
        result, min_x, min_y = create_mask_for_image(
            svs_path,
            analysis_parameters['analysis_region_json'],
            str(mask_save_path)
        )

        # Wykonanie predykcji
        logger.info("Creating prediction")
        prediction = engine.make_prediction(
            svs_path=svs_path,
            location=[min_x, min_y],
            size=[result.shape[1], result.shape[0]],
            save_path=str(save_path),
            save_dir=str(analysis_folder),
            on_gpu=on_gpu
        )

        # Wycinanie maski z predykcji
        logger.info("Cutting mask from prediction")
        triangle = cut_mask_from_array(
            prediction,
            analysis_parameters['analysis_region_json'],
            min_x,
            min_y
        )

        end_time = time.time()
        prediction_time = end_time - start_time
        json_data["prediction_time"] = prediction_time

        # Nakładanie predykcji na obraz
        start_time = time.time()
        logger.info("Overlaying prediction on image")
        result_path = engine.overlay_png_with_pred(
            svs_path=svs_path,
            overlay=triangle,
            save_path=str(results_folder),
            location=[min_x, min_y]
        )

        end_time = time.time()
        overlay_time = end_time - start_time

        # Aktualizacja danych JSON
        json_data.update({
            "overlay_time": overlay_time,
            "status": "finished",
            "result_image_path": result_path
        })

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        json_data.update({
            "status": "error",
            "error_message": str(e)
        })

    finally:
        # Zapisz końcowy status
        with open(json_file_path, 'w') as file:
            json.dump(json_data, file)

        # Wywołaj callback
        try:
            logger.info(f"Calling results ready callback for analysis_id: {analysis_id}")
            asyncio.run(call_results_ready(analysis_id, api_url))
        except Exception as e:
            logger.error(f"Error in callback: {str(e)}")

    logger.info(f"Analysis completed: {analysis_id}")
    return json_data