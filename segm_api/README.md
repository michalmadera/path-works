# Segmentation API

## Environment Preparation
In the selected localization, create two folders for data and results, for example:
mkdir DATA

## Running Container
To run the container, execute the following command:
docker run -v "$(pwd)/DATA:/DATA" -v "$(pwd)/RESULTS:/RESULTS" -p 8000:8000 --gpus all -d segm_api

Two folders are mapped: DATA to DATA and RESULTS to RESULTS. Additionally, port 8000 is mapped to 8000, and the `--gpus all` flag ensures the utilization of all available GPUs.

## analyzeWSI Endpoint
To analyze your image, place it in the data folder. Then, to make a request to the analyzeWSI endpoint, use, for example, Postman. Select the POST method and provide the following parameters:
- "svs_path": "DATA/test.jpg"
- "analysis_type": 1

In the request body, specify:
{
    "analysis_region": [0, 0],
    "is_normalized": false
}

The function will return an analysis_id of type string:
"0"

To download an example image, use the following link: [Example Image](https://tiatoolbox.dcs.warwick.ac.uk/sample_imgs/breast_tissue.jpg)

## resultReady Endpoint
The resultReady endpoint takes an analysis_id as input and returns a JSON file with result data. In Postman, select the GET method and provide the following parameter:
- "analysis_id": 0

The endpoint returns:
{
"analysis_id": "0",
"svs_path": "svs_dir/test.jpg",
"analysis_type": "1",
"region": "(0, 0)",
"is_normalized": "False",
"status": "finished",
"result_json_path": "RESULTS/0/0.json",
"result_image_path": "RESULTS/0/result.tif"
}