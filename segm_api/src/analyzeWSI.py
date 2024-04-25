from fastapi import FastAPI, HTTPException
import json
import asyncio

app = FastAPI()

analysis_db = {}
results_db = {}


@app.post("/analyzeWSI")
async def analyze_wsi(svs_path: str, analysis_type: str, analysis_parameters_json: str) -> str:
    analysis_parameters = json.loads(analysis_parameters_json)

    analysis_region = analysis_parameters.get("analysis_region", "cały obraz")
    is_normalized = analysis_parameters.get("is_normalized", False)

    analysis_id = len(analysis_db) + 1
    analysis_db[analysis_id] = {
        "svs_path": svs_path,
        "analysis_type": analysis_type,
        "analysis_region": analysis_region,
        "is_normalized": is_normalized
    }
    return {"analysis_id": analysis_id}


@app.post("/resultsReady")
async def results_ready(analysis_id: str, result_file_path: str, results_data: dict ) -> dict:
    analysis_id = analysis_id
    results_db[analysis_id] = {
        "analysis_id": analysis_id,
        "result_file_path": result_file_path,
        "results_data": results_data
    }
    return {"analysis_id": analysis_id, "result_file_path": result_file_path, "results_data": results_data}


@app.get("/analysis/{analysis_id}")
async def get_analysis_info(analysis_id: int) -> dict:
    if analysis_id not in analysis_db:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_db[analysis_id]


async def initialize_results():
    sample_results = [
        {
            "analysis_id": "1",
            "result_file_path": "/path/to/result1.txt",
            "results_data": {"result1": 10, "result2": 20}
        },
        {
            "analysis_id": "2",
            "result_file_path": "/path/to/result2.txt",
            "results_data": {"result3": 30, "result4": 40}
        },
        {
            "analysis_id": "3",
            "result_file_path": "/path/to/result3.txt",
            "results_data": {"result5": 50, "result6": 60}
        }
    ]

    for result_data in sample_results:
        analysis_id = result_data["analysis_id"]
        result_file_path = result_data["result_file_path"]
        results_data = result_data["results_data"]
        await results_ready(analysis_id=analysis_id, result_file_path=result_file_path, results_data=results_data)


@app.get("/results/{analysis_id}")
async def get_results_info(analysis_id: str) -> dict:
    if analysis_id not in results_db:
        raise HTTPException(status_code=404, detail="Results not found")
    return results_db[analysis_id]


@app.get("/results")
async def get_all_results() -> dict:
    return results_db

@app.get("/analysis")
async def get_all_analysis() -> dict:
    return analysis_db

async def initialize_analysis():
    sample_analyses = [
        {
            "svs_path": "path/to/image1.svs",
            "analysis_type": "type1",
            "analysis_parameters_json": '{"analysis_region": "cały obraz", "is_normalized": false}'
        },
        {
            "svs_path": "path/to/image2.svs",
            "analysis_type": "type2",
            "analysis_parameters_json": '{"analysis_region": "obszar 1", "is_normalized": true}'
        },
        {
            "svs_path": "path/to/image3.svs",
            "analysis_type": "type3",
            "analysis_parameters_json": '{"analysis_region": "obszar 2", "is_normalized": false}'
        }
    ]

    for analysis_data in sample_analyses:
        svs_path = analysis_data["svs_path"]
        analysis_type = analysis_data["analysis_type"]
        analysis_parameters_json = analysis_data["analysis_parameters_json"]
        await analyze_wsi(svs_path=svs_path, analysis_type=analysis_type,
                          analysis_parameters_json=analysis_parameters_json)


if __name__ == "__main__":
    import uvicorn

    asyncio.run(initialize_analysis())
    asyncio.run(initialize_results())

    uvicorn.run(app, host="0.0.0.0", port=8000)
