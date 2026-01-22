# 1. Das Fundament: Maximale CUDA Power
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

SHELL ["/bin/bash","-lc"]

# 2. System-Setup & Python 3.12 Installation (WICHTIG: Hat gefehlt!)
# Wir holen uns Python 3.12 via PPA, da Ubuntu 22.04 standardmäßig 3.10 hat.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Berlin \
    HF_HOME=/workspace/.cache/hf \
    TRANSFORMERS_CACHE=/workspace/.cache/hf/transformers \
    HF_HUB_CACHE=/workspace/.cache/hf/hub \
    PATH="/home/root/.local/bin:${PATH}" \
# FIX: PYTHONPATH sauber ohne Selbstreferenz setzen
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

# Pip für Python 3.12 installieren (da es nicht automatisch dabei ist)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Python 3.12 als festen Standard definieren
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && ln -sf /usr/bin/python3.12 /usr/bin/python

# 3. Das Herzstück: PyTorch 2.7 (Stabil für CUDA 12.8)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128


# 4. Flash Attention (Die stabilste High-Performance Version)
# Wir installieren die neueste Version von flash-attn. 
# Diese nutzt auf Hopper-GPUs automatisch die besten Kernel und ist für LTX-2 perfekt.
RUN python -m pip install ninja packaging wheel
RUN python -m pip install --no-cache-dir flash-attn --no-build-isolation

WORKDIR /workspace

# 6. Restliche Python-Deps aus requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN python -V && python -m pip -V \
 && python -m pip install --no-cache-dir -r /tmp/requirements.txt

# Nichts weiter – start.sh kümmert sich um Clone, Modelle, Jupyter etc.
COPY . .
RUN chmod +x /workspace/start.sh \
    && chmod +x /workspace/init.sh \
    && chmod +x /workspace/logs.sh

EXPOSE 8888 8000
CMD ["/bin/bash","-lc","/workspace/start.sh"]
