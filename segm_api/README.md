# Segmentation API

## Environment Preparation
In the selected localization, create two folders for data and results, for example:
mkdir DATA

## Pull containers
Pull both containers: jkuzn/segm_api-web and jkuzn/segm_api-celery_worker.

## Prepare docker-compose and .env file
Download docker-compose.yml from github and create .env file that looks like below:
- DATA_VOLUME=/path/to/DATA/folder
- RESULTS_VOLUME=/path/to/RESULTS/folder 
- ON_GPU=FALSE
- SHM_SIZE=8G 

## Running the container
To run container make sure that docker-compose.yml and .env are in the same place and run belowe command:
- docker-compose up

## analyzeWSI Endpoint
To analyze your image, place it in the data folder. Then, make a request to the analyzeWSI endpoint. For example, using Postman, select the POST method and provide the following JSON body:
{
  "svs_path": "/DATA/test.jpg",
  "analysis_type": 1,
  "analysis_parameters": {
    "analysis_region_json": "/DATA/0.json",
    "is_normalized": false
  }
}
The function will return an analysis_id of type string:
{
    "task_id": "3bc182f6-161f-45e4-8053-6bbef9e6cfb4"
}

To download an example image, use the following link: [Example Image](https://tiatoolbox.dcs.warwick.ac.uk/sample_imgs/breast_tissue.jpg)
Example region json:
{
  "type": null,
  "id": null,
  "geometry": {
    "type": null,
    "coordinates": [
      {
        "x": 300,
        "y": 400
      },
      {
        "x": 1000,
        "y": 2500
      },
      {
        "x": 1200,
        "y": 1835
      },
      {
        "x": 1045,
        "y": 2264
      },
      {
        "x": 1394,
        "y": 1304
      }
    ]
  }
}

## resultReady Endpoint
The resultsReady endpoint takes an analysis_id as input and returns a JSON file with result data. In Postman, select the GET method and provide the following parameter in the URL path:
/resultsReady/{analysis_id}


The endpoint returns:
{
"analysis_id": "3bc182f6-161f-45e4-8053-6bbef9e6cfb4",
"svs_path": "/DATA/test.jpg",
"analysis_type": "1",
"region": "(0, 0)",
"is_normalized": "False",
"status": "finished",
"result_json_path": "/RESULTS/0/0.json",
"result_image_path": "/RESULTS/0/result.tif"
}

## checkStatus Endpoint
The checkStatus endpoint takes an analysis_id as input and returns analysis status. In Postman, select the GET method and provide the following parameter in the URL path:
/checkStatus/{analysis_id}

The endpoint returns:
"in_process"