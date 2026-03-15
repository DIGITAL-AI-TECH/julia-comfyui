FROM runpod/worker-comfyui:5.7.1-base

# ComfyUI_IPAdapter_plus — sem insightface (evita conflito com onnxruntime-gpu da base image)
# Habilita: IPAdapterAdvanced, IPAdapterModelLoader, CLIPVisionLoader
# Desabilitado: InsightFaceLoader, IPAdapterFaceIDPlus (requer insightface — add depois)
RUN git clone --quiet https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus

# Symlink: /comfyui/models/ipadapter → /runpod-volume/models/ipadapter (montado em runtime)
# IPAdapterModelLoader busca em folder_paths.models_dir/ipadapter = /comfyui/models/ipadapter
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/ipadapter /comfyui/models/ipadapter

# Verificação no build
RUN ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && \
    echo "IPAdapter OK"
