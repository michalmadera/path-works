FROM python:3.10

WORKDIR /segm-api

COPY requirements.txt .
COPY ./src ./src

RUN pip install -r requirements.txt

CMD ["python", "./src/analyzeWSI.py"]

