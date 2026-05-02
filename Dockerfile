# 1. Das Fundament: Maximale CUDA Power
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

SHELL ["/bin/bash","-lc"]

# 2. System-Setup & Python 3.12
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Berlin \
    HF_HOME=/workspace/.cache/hf \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:/opt/venv/bin:/root/.local/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    PYTHONPATH="/workspace/LTX-2/packages/ltx-core/src:/workspace/LTX-2/packages/ltx-pipelines/src" \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    python-is-python3 \
    git \
    ffmpeg \
    curl \
    ca-certificates \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
  && python -m pip install --upgrade pip setuptools wheel \
  && python --version \
  && python -m pip --version

# 3. PyTorch 2.7 für CUDA 12.8
RUN python -m pip install --no-cache-dir \
      "torch==2.7.0+cu128" \
      "torchvision==0.22.0+cu128" \
      "torchaudio==2.7.0+cu128" \
      --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install --no-cache-dir ninja packaging psutil pybind11 qwen_tts einops

RUN python -m pip install --no-cache-dir "flash-attn==2.8.3" --no-build-isolation \
 && python -c "import flash_attn; print('flash_attn ok', flash_attn.__version__)"

WORKDIR /workspace

# 6. Restliche Python-Deps
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# 7. Scripte & Start
COPY . .
COPY config/jupyter_server_config.py /usr/local/etc/jupyter/jupyter_server_config.py

EXPOSE 8888 8000
CMD ["/bin/bash","-lc","bash /workspace/start.sh"]