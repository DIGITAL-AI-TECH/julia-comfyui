FROM runpod/worker-comfyui:main-base

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

# ComfyUI_FluxMod: ChromaDiffusionLoader + ChromaPaddingRemoval (suporte ao modelo Chroma)
# Sem requirements.txt — usa apenas deps do base image (torch etc.)
# DIAGNÓSTICO: gonzalomoChroma usa arquitetura "Chroma" (FLUX sem time_in).
#   CheckpointLoaderSimple → KSampler falha com: 'Chroma' object has no attribute 'time_in'
#   Solução: ChromaDiffusionLoader carrega o modelo de forma compatível com KSampler
RUN git clone --quiet https://github.com/lodestone-rock/ComfyUI_FluxMod.git \
    /comfyui/custom_nodes/ComfyUI_FluxMod

# ─── Patch pulidflux.py: lazy imports para evitar startup hang ────────────────
# PROBLEMA: pulidflux.py importa FaceAnalysis, FaceRestoreHelper, init_parsing_model
#   no nível de MÓDULO (linhas 12-14), mas só usa dentro de métodos (linhas 281/374/384).
#   Essas importações executam código pesado de GPU/modelo no startup do ComfyUI,
#   causando o container ficar em initializing por 45+ minutos.
# SOLUÇÃO: mover para lazy imports (function-level) — importar só quando o método executa.
RUN python3 << 'EOF'
import re

path = '/comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py'
with open(path, 'r') as f:
    content = f.read()

# 1. Remove top-level imports (comment them out)
content = content.replace(
    'from insightface.app import FaceAnalysis\n',
    '# lazy-import: FaceAnalysis imported inside load_insightface()\n'
)
content = content.replace(
    'from facexlib.parsing import init_parsing_model\n',
    '# lazy-import: init_parsing_model imported inside execute()\n'
)
content = content.replace(
    'from facexlib.utils.face_restoration_helper import FaceRestoreHelper\n',
    '# lazy-import: FaceRestoreHelper imported inside execute()\n'
)

# 2. Add lazy import before FaceAnalysis usage in load_insightface()
content = content.replace(
    '        model = FaceAnalysis(name="antelopev2"',
    '        from insightface.app import FaceAnalysis\n        model = FaceAnalysis(name="antelopev2"'
)

# 3. Add lazy imports before FaceRestoreHelper usage in execute()
content = content.replace(
    '        face_helper = FaceRestoreHelper(\n',
    '        from facexlib.parsing import init_parsing_model\n        from facexlib.utils.face_restoration_helper import FaceRestoreHelper\n        face_helper = FaceRestoreHelper(\n'
)

with open(path, 'w') as f:
    f.write(content)

# Verify patch was applied
assert '# lazy-import: FaceAnalysis' in content, 'FaceAnalysis patch failed'
assert '# lazy-import: init_parsing_model' in content, 'init_parsing_model patch failed'
assert '# lazy-import: FaceRestoreHelper' in content, 'FaceRestoreHelper patch failed'
assert 'from insightface.app import FaceAnalysis\n        model = FaceAnalysis' in content, 'lazy FaceAnalysis injection failed'
assert 'from facexlib.parsing import init_parsing_model\n        from facexlib.utils.face_restoration_helper import FaceRestoreHelper\n        face_helper = FaceRestoreHelper' in content, 'lazy facexlib injection failed'
print('pulidflux.py patched OK — all 3 lazy imports verified')
EOF

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
    ln -sf /runpod-volume/models/clip         /comfyui/models/clip         && \
    ln -sf /runpod-volume/models/checkpoints  /comfyui/models/diffusion_models
# NOTA: diffusion_models → checkpoints para ChromaDiffusionLoader encontrar gonzalomoChroma_v30.safetensors

# ─── Verificação de build ──────────────────────────────────────────────────────
RUN /opt/venv/bin/python -c "import onnxruntime, insightface, timm, facexlib, einops, kornia; print('onnxruntime ' + onnxruntime.__version__); print('providers: ' + str(onnxruntime.get_available_providers())); print('insightface OK'); print('timm OK'); print('facexlib OK')" && \
    ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && echo "IPAdapter OK" && \
    ls /comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py && echo "PuLID OK" && \
    ls /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/nodes.py && echo "FluxMod/ChromaWrapper OK"
