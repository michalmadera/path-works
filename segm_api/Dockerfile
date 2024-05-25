FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Instalacja zależności systemowych, Pythona 3.10 i narzędzi kompilacyjnych
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

# Instalacja PIP za pomocą get-pip.py
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.10 get-pip.py

# Ustawienie Pythona 3.10 jako domyślnego
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /segm-api

COPY requirements.txt .
COPY ./src ./src
COPY ./analysis ./analysis
COPY README.md README.md
COPY start_celery.sh ./start_celery.sh

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Instalacja PyTorch z obsługą CUDA
RUN python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalacja pozostałych zależności
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt --default-timeout=100 future

# Ustawienie uprawnień dla start_celery.sh
RUN chmod +x ./start_celery.sh

# Komenda startowa
CMD ["uvicorn", "src.analyzeWSI:app", "--host", "0.0.0.0", "--port", "8000"]
