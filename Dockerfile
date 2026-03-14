FROM runpod/worker-comfyui:5.7.1-base

# g++ necessário para compilar insightface do source (Python 3.12 — sem wheel cp312)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends g++ cmake python3-dev && \
    rm -rf /var/lib/apt/lists/*

# InsightFace + onnxruntime — usa /opt/venv/bin/pip (venv do worker em runtime)
RUN /opt/venv/bin/pip install --quiet --no-cache-dir onnxruntime insightface==0.7.3

# ComfyUI_IPAdapter_plus (sem requirements.txt no repo — deps já instaladas acima)
RUN git clone --quiet https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus

# Symlinks: /comfyui/models/ipadapter → /runpod-volume/models/ipadapter (runtime)
# IPAdapterModelLoader usa folder_paths.models_dir/ipadapter = /comfyui/models/ipadapter
# extra_model_paths.yaml pode não cobrir tipos customizados; symlink é mais confiável
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/ipadapter /comfyui/models/ipadapter && \
    ln -sf /runpod-volume/models/insightface /comfyui/models/insightface

# Também tenta via extra_model_paths.yaml como fallback
RUN printf '\ncomfyui_ipadapter:\n    base_path: /runpod-volume/\n    ipadapter: models/ipadapter/\n' \
    >> /comfyui/extra_model_paths.yaml

# Verificação no build — falha aqui = imagem não sobe com dependência quebrada
RUN /opt/venv/bin/python -c "import insightface; print('InsightFace OK')" && \
    ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && \
    echo "IPAdapter OK"
