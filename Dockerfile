FROM runpod/worker-comfyui:5.7.1-base

# Build tools para compilar insightface (sem wheel cp312)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends g++ cmake python3-dev && \
    rm -rf /var/lib/apt/lists/*

# insightface SEM instalar onnxruntime explicitamente
# A base image já tem onnxruntime-gpu que satisfaz o requisito >=1.9
# Instalar onnxruntime (CPU) separadamente sobrescreve onnxruntime-gpu → crash
RUN pip install --quiet --no-cache-dir insightface==0.7.3

# ComfyUI_IPAdapter_plus — habilita InsightFaceLoader, IPAdapterFaceIDPlus, IPAdapterAdvanced
RUN git clone --quiet https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus

# Symlinks: resolve modelos do volume network em runtime
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/ipadapter /comfyui/models/ipadapter && \
    ln -sf /runpod-volume/models/insightface /comfyui/models/insightface

# Verificação de build
RUN python3 -c "import insightface; print('InsightFace OK')" && \
    ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && \
    echo "IPAdapter OK"
