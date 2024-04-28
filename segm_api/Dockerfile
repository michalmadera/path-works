FROM python:3.10

WORKDIR /segm-api

COPY requirements.txt .
COPY ./src ./src
COPY ./analysis ./analysis
COPY README.md README.md

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y libvips
RUN pip install -r requirements.txt

CMD ["python", "./src/analyzeWSI.py"]

