FROM runpod/worker-comfyui:5.7.1-base

# Build tools para compilar insightface do source (sem wheel cp312)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends g++ cmake python3-dev && \
    rm -rf /var/lib/apt/lists/*

# onnxruntime-gpu + insightface — instalar na VENV do ComfyUI (/opt/venv)
# CRÍTICO: usar /opt/venv/bin/pip (não pip do sistema)
# onnxruntime-gpu==1.17.3: suporta CUDA 11.8–12.x + cuDNN 8.x
# Versões 1.19+ requerem cuDNN 9 — incompatível com base image (CUDA 12.4, cuDNN 8)
RUN /opt/venv/bin/pip install --quiet --no-cache-dir \
    "onnxruntime-gpu==1.17.3" \
    "insightface==0.7.3"

# ComfyUI_IPAdapter_plus — habilita InsightFaceLoader + IPAdapterFaceIDPlus
RUN git clone --quiet https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus

# Symlinks: modelos do volume network → paths do ComfyUI (resolvidos em runtime)
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/ipadapter /comfyui/models/ipadapter && \
    ln -sf /runpod-volume/models/insightface /comfyui/models/insightface

# Verificação na VENV — debug de versão + import check
# NOTA: python -c deve ser em linha única (BuildKit não aceita aspas duplas multiline no RUN)
RUN /opt/venv/bin/python -c "import onnxruntime, insightface; print('onnxruntime ' + onnxruntime.__version__); print('providers: ' + str(onnxruntime.get_available_providers())); print('insightface OK')" && \
    ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && \
    echo "IPAdapter OK"
