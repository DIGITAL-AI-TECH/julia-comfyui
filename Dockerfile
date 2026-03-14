FROM runpod/worker-comfyui:5.7.1-base

# InsightFace — wheel pré-compilado para CP311 Linux (evita build de 15min)
# Fallback para pip install caso o wheel mude de URL
RUN pip install --quiet --no-cache-dir onnxruntime && \
    pip install --quiet --no-cache-dir \
    https://github.com/Gourieff/Assets/releases/download/model-pack-anime-v1/insightface-0.7.3-cp311-cp311-linux_x86_64.whl || \
    pip install --quiet --no-cache-dir insightface==0.7.3

# ComfyUI_IPAdapter_plus
RUN git clone --quiet https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus && \
    pip install --quiet --no-cache-dir \
    -r /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/requirements.txt

# Verificação no build — falha aqui = imagem não sobe com dependência quebrada
RUN python3 -c "import insightface; print('InsightFace OK')" && \
    ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && \
    echo "IPAdapter OK"
