FROM nvidia/cuda:12.4.0-base-ubuntu20.04


RUN apt-get update && apt-get install -y software-properties-common wget


RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libvips \
    libopenslide0 \
    libopenslide-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.10 get-pip.py

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /segm-api

COPY requirements.txt .
COPY ./src ./src
COPY ./analysis ./analysis
COPY README.md README.md
COPY start_celery.sh ./start_celery.sh

RUN python3.10 -m pip install --upgrade pip

RUN python3.10 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN python3.10 -m pip install --no-cache-dir redis>=4.5.0

RUN python3.10 -m pip install --no-cache-dir -r requirements.txt --default-timeout=100 future

RUN chmod +x ./start_celery.sh

CMD ["uvicorn", "src.analyzeWSI:app", "--host", "0.0.0.0", "--port", "8000"]