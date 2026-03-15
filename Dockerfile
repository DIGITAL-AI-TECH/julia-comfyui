FROM runpod/worker-comfyui:5.7.1-base

# ─── Build tools ───────────────────────────────────────────────────────────────
# Necessários para compilar insightface do source (sem wheel cp312)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends g++ cmake python3-dev && \
    rm -rf /var/lib/apt/lists/*

# ─── Python deps — instalar SEMPRE na VENV (/opt/venv) ────────────────────────
# CRÍTICO: usar /opt/venv/bin/pip — pip do sistema não serve para o ComfyUI runtime
# onnxruntime-gpu: OBRIGATÓRIO — CPU version crasha container (conflito libs GPU)
# NÃO instalar open_clip_torch: base image já tem! Reinstalar quebra ComfyUI startup
# timm, ftfy: deps do eva_clip (PuLID) não presentes no base
# facexlib --no-deps: CRÍTICO — facexlib deps padrão puxam numba que faz compilação
#   CUDA JIT na primeira importação (10-30 min), travando o startup do ComfyUI
# einops, kornia: deps do ComfyUI-PuLID-Flux-Enhanced (leves, sem CUDA init)
RUN /opt/venv/bin/pip install --quiet --no-cache-dir \
    onnxruntime-gpu \
    insightface==0.7.3 \
    timm \
    ftfy \
    einops \
    kornia && \
    /opt/venv/bin/pip install --quiet --no-cache-dir --no-deps facexlib && \
    /opt/venv/bin/pip install --quiet --no-cache-dir filterpy scipy

# ─── Custom Nodes ──────────────────────────────────────────────────────────────
# IPAdapter Plus: IPAdapterFaceID + IPAdapterInsightFaceLoader
RUN git clone --quiet https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus

# PuLID Flux Enhanced: PulidFluxModelLoader + ApplyPulidFlux
RUN git clone --quiet https://github.com/sipie800/ComfyUI-PuLID-Flux-Enhanced.git \
    /comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced

# ─── Symlinks: Network Volume → ComfyUI model paths ───────────────────────────
# Modelos ficam no volume persistente (/runpod-volume/models/)
# ComfyUI procura em /comfyui/models/ → symlinks resolvem em runtime
RUN mkdir -p /comfyui/models && \
    ln -sf /runpod-volume/models/ipadapter    /comfyui/models/ipadapter    && \
    ln -sf /runpod-volume/models/insightface  /comfyui/models/insightface  && \
    ln -sf /runpod-volume/models/pulid        /comfyui/models/pulid        && \
    ln -sf /runpod-volume/models/eva_clip     /comfyui/models/eva_clip     && \
    ln -sf /runpod-volume/models/unet         /comfyui/models/unet         && \
    ln -sf /runpod-volume/models/vae          /comfyui/models/vae          && \
    ln -sf /runpod-volume/models/clip         /comfyui/models/clip

# ─── Verificação de build ──────────────────────────────────────────────────────
RUN /opt/venv/bin/python -c "import onnxruntime, insightface, timm, facexlib, einops, kornia; print('onnxruntime ' + onnxruntime.__version__); print('providers: ' + str(onnxruntime.get_available_providers())); print('insightface OK'); print('timm OK'); print('facexlib OK')" && \
    ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && echo "IPAdapter OK" && \
    ls /comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py && echo "PuLID OK"
