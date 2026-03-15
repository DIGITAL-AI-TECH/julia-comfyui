FROM runpod/worker-comfyui:5.7.1-base

# Build tools para compilar insightface do source (sem wheel cp312)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends g++ cmake python3-dev && \
    rm -rf /var/lib/apt/lists/*

# onnxruntime-gpu + insightface — instalar na VENV do ComfyUI (/opt/venv)
# CRÍTICO: usar /opt/venv/bin/pip (não pip do sistema)
# NOTA: insightface usa lazy import — node IPAdapterInsightFaceLoader registra sem insightface,
#        mas a insightface é necessária em runtime quando o node executa
RUN /opt/venv/bin/pip install --quiet --no-cache-dir onnxruntime-gpu insightface==0.7.3

# ComfyUI_IPAdapter_plus — habilita IPAdapterInsightFaceLoader + IPAdapterFaceID
RUN git clone --quiet https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus

# Symlinks: modelos do volume network → paths do ComfyUI (resolvidos em runtime)
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/ipadapter /comfyui/models/ipadapter && \
    ln -sf /runpod-volume/models/insightface /comfyui/models/insightface

# Verificação na VENV (não no python3 do sistema — resultado enganoso)
RUN /opt/venv/bin/python -c "import onnxruntime, insightface; print('onnxruntime ' + onnxruntime.__version__); print('providers: ' + str(onnxruntime.get_available_providers())); print('insightface OK')" && \
    ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && \
    echo "IPAdapter OK"
