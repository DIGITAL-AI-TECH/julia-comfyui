FROM runpod/worker-comfyui:5.7.1-base

# Build tools para compilar insightface do source (sem wheel cp312)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends g++ cmake python3-dev && \
    rm -rf /var/lib/apt/lists/*

# onnxruntime-gpu + insightface
# - onnxruntime (CPU) crasha o container ao substituir onnxruntime-gpu do base image
# - a venv não tem onnxruntime nativa → insightface falha sem ele
# - solução: instalar onnxruntime-gpu explicitamente (preserva suporte a CUDA)
RUN pip install --quiet --no-cache-dir onnxruntime-gpu insightface==0.7.3

# ComfyUI_IPAdapter_plus — habilita InsightFaceLoader + IPAdapterFaceIDPlus
RUN git clone --quiet https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus

# Symlinks: modelos do volume network → paths do ComfyUI (resolvidos em runtime)
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/ipadapter /comfyui/models/ipadapter && \
    ln -sf /runpod-volume/models/insightface /comfyui/models/insightface

# Verificação de build
RUN python3 -c "import insightface, onnxruntime; print(f'InsightFace OK | onnxruntime {onnxruntime.__version__}')" && \
    ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && \
    echo "IPAdapter OK"
