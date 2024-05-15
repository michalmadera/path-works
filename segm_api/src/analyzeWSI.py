import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from .tasks import perform_analysis
import json
app = FastAPI()

class AnalysisParameters(BaseModel):
    analysis_region_json: str
    is_normalized: bool

@app.post("/analyzeWSI")
async def analyze_wsi(svs_path: str, analysis_type: int, analysis_parameters: AnalysisParameters, background_tasks: BackgroundTasks):
    analysis_parameters = analysis_parameters.dict()
    svs_path = svs_path.strip('"')

    results_dir = "/RESULTS"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    folders = os.listdir(results_dir)
    num_folders = len(folders)
    analysis_id = str(num_folders)

    print(f"Wysyłanie zadania Celery: svs_path={svs_path}, analysis_type={analysis_type}, analysis_parameters={analysis_parameters}, analysis_id={analysis_id}")
    task = perform_analysis.delay(svs_path, analysis_type, analysis_parameters, analysis_id)
    print(f"Zadanie Celery wysłane: task_id={task.id}")

    return {"analysis_id": analysis_id, "task_id": task.id}

@app.get("/checkStatus")
async def check_status(analysis_id: str):
    file_path = find_results_path("/RESULTS", analysis_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File for analysis ID {analysis_id} not found")

    try:
        with open(file_path, "r") as file:
            results_data = json.load(file)
    except IOError:
        raise HTTPException(status_code=503, detail="Unable to read status file, try again later.")

    status = results_data['status']
    return {"status": status}

@app.get("/resultsReady/{analysis_id}")
async def results_ready(analysis_id: str):
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
    uvicorn.run("analyzeWSI:app", host="0.0.0.0", port=8000)
