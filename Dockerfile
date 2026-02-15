# 1. Das Fundament: Maximale CUDA Power
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash","-lc"]




# 2. System-Setup & Python 3.12
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Berlin \
    HF_HOME=/workspace/.cache/hf \
    # FIX: Pfade für CUDA Compiler setzen
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:/root/.local/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    PYTHONPATH="/workspace/LTX-2/packages/ltx-core/src:/workspace/LTX-2/packages/ltx-pipelines/src"

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    ffmpeg \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pip für Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && ln -sf /usr/bin/python3.12 /usr/bin/python

# 3. PyTorch 2.7 (Stabil für CUDA 12.8)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128


RUN python -m pip install --no-cache-dir -U pip setuptools wheel \
 && python -m pip install --no-cache-dir ninja packaging psutil pybind11 einops

RUN python -m pip install --no-cache-dir "flash-attn==2.8.3" --no-build-isolation \
 && python -c "import flash_attn; print('flash_attn ok', flash_attn.__version__)"



WORKDIR /workspace

# 6. Restliche Python-Deps
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# 7. Scripte & Start
COPY . .



EXPOSE 8888 8000
CMD ["/bin/bash","-lc","/workspace/start.sh"]