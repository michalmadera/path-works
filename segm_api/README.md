# Segmentation API

## Environment Preparation
In the selected localization, create two folders for data and results, for example:
mkdir DATA

## Nvidia Tools Installation
To enable GPU support, you need to install the NVIDIA Container Toolkit. Follow these steps:

Update package lists and install dependencies:
- sudo apt-get update
- sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

Add the NVIDIA package repositories:
- curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
- distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
- curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

Install the NVIDIA Container Toolkit:
- sudo apt-get update
- sudo apt-get install -y nvidia-docker2
- sudo systemctl restart docker

Test your installation
- docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu20.04 nvidia-smi

## Prepare docker-compose and .env file
Download docker-compose.yml from github and create .env file that looks like below:
- DATA_VOLUME=/path/to/DATA/folder
- RESULTS_VOLUME=/path/to/RESULTS/folder 
- ON_GPU=FALSE
- SHM_SIZE=8G 
- results_ready_callback_url=http://localhost:8000/resultsReadyCallbackTester/

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
    "analysis_id": "3bc182f6-161f-45e4-8053-6bbef9e6cfb4"
}

To download an example image, use the following link: [Example Image](https://tiatoolbox.dcs.warwick.ac.uk/sample_imgs/breast_tissue.jpg), [Svs example image](https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/wsi4_12k_12k.svs), [Large image example](https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/CMU-1.ndpi)
Example region json:
{ "type": "FeatureCollection", "features": [ { "type": "Feature", "id": "26cada93-5f19-4709-aa2b-d445a78dfb2c", "geometry": { "type": "Polygon", "coordinates": [ [ [300, 400], [1000, 2500], [1200, 1835], [1045, 2264], [1394, 1304] ] ] }, "properties": { "objectType": "annotation" } } ] }

## checkStatus Endpoint
The checkStatus endpoint takes an analysis_id as input and returns a JSON file with result data. In Postman, select the GET method and provide the following parameter in the URL path:
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

## Callbacks
The system uses callbacks to notify when the analysis is complete. The callback URL is specified in the results_ready_callback_url environment variable. The call_results_ready function sends a POST request to this URL with the analysis_id in the JSON body.

