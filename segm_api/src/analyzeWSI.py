from fastapi import FastAPI, HTTPException
import json
import asyncio
import segmentation_engine as engine
import os
from pydantic import BaseModel, Field
from typing import List

app = FastAPI()


class AnalysisParameters(BaseModel):
    analysis_region: List[int]
    is_normalized: bool


class AnalysisRequest(BaseModel):
    svs_path: str
    analysis_type: str
    analysis_parameters: AnalysisParameters


@app.post("/analyzeWSI")
async def analyze_wsi(svs_path: str, analysis_type: str, analysis_parameters_json: AnalysisParameters) -> str:
    analysis_parameters = analysis_parameters_json.dict()

    analysis_region = analysis_parameters["analysis_region"]
    is_normalized = analysis_parameters["is_normalized"]

    svs_path = svs_path.strip('"')

    analysis_dir = "analysis"
    results_dir = "/RESULTS"

    folders = os.listdir(results_dir)
    num_folders = len(folders)

    analysis_folder = os.path.join(analysis_dir, f"{num_folders}")
    results_folder = os.path.join(results_dir, f"{num_folders}")
    os.makedirs(analysis_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    save_path = os.path.join(analysis_folder, "sample.jpg")
    json_file_path = os.path.join(results_folder, f"{num_folders}.json")

    json_data = {"analysis_id": f"{num_folders}", "svs_path": f"{svs_path}",
                 "analysis_type": f"{analysis_type}", "region": f"{analysis_region[0], analysis_region[1]}",
                 "is_normalized": f"{is_normalized}", "status": "in process",
                 "result_json_path": f"{json_file_path}",}
    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file)

    try:
        prediction = engine.make_prediction(svs_path=svs_path, location=[analysis_region[0], analysis_region[1]],
                                            size=[512, 512], save_path=save_path, save_dir=analysis_folder)

    except Exception as e:
        json_data["status"] = "error"
        json_data["status_message"] = str(e)

    if json_data["status"] == "in process":
        try:
            engine.overlay_tif_with_pred(svs_path=svs_path, overlay=prediction, save_path=results_folder,
                                         location=[analysis_region[0], analysis_region[1]])
            json_data["status"] = "finished"
            json_data["result_image_path"] = f"{results_folder}/result.tif"
        except Exception as e:
            json_data["status"] = "error"
            json_data["status_message"] = str(e)

    with open(json_file_path, "w") as json_file:
        json.dump(json_data, json_file)

    print(json_data)

    return f"{num_folders}"


@app.get("/resultsReady")
async def results_ready(analysis_id: str) -> dict:

    results_dir = "/RESULTS"
    results_dir = os.path.join(results_dir, f"{analysis_id}")
    file_path = os.path.join(results_dir, f"{analysis_id}.json")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File for analysis ID {analysis_id} not found")

    with open(file_path, "r") as file:
        results_data = json.load(file)

    return results_data


# @app.get("/analysis/{analysis_id}")
# async def get_analysis_info(analysis_id: int) -> dict:
#     if analysis_id not in analysis_db:
#         raise HTTPException(status_code=404, detail="Analysis not found")
#     return analysis_db[analysis_id]
@app.get("/results/{result_id}")
async def get_result_info(result_id: int) -> dict:
    result_file_path = f"RESULTS/{result_id}.json"
    try:
        with open(result_file_path, "r") as json_file:
            result_data = json.load(json_file)
        return result_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Result not found")


# @app.get("/RESULTS")
# async def get_all_results() -> dict:
#     return results_db
#
# @app.get("/analysis")
# async def get_all_analysis() -> dict:
#     return analysis_db


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
