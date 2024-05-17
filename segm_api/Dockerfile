FROM python:3.10

WORKDIR /segm-api

COPY requirements.txt .
COPY ./src ./src
COPY ./analysis ./analysis
COPY README.md README.md
COPY start_celery.sh ./start_celery.sh

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get update && apt-get install -y libvips
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --default-timeout=100 future

RUN chmod +x ./start_celery.sh

CMD ["uvicorn", "src.analyzeWSI:app", "--host", "0.0.0.0", "--port", "8000"]
