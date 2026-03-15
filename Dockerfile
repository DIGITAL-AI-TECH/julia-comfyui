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
RUN git clone --quiet --depth 1 https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
    /comfyui/custom_nodes/ComfyUI_IPAdapter_plus

# PuLID Flux Enhanced: PulidFluxModelLoader + ApplyPulidFlux
RUN git clone --quiet --depth 1 https://github.com/sipie800/ComfyUI-PuLID-Flux-Enhanced.git \
    /comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced

# ComfyUI_FluxMod: ChromaDiffusionLoader + ChromaPaddingRemoval (suporte ao modelo Chroma)
# Sem requirements.txt — usa apenas deps do base image (torch etc.)
# DIAGNÓSTICO: gonzalomoChroma usa arquitetura "Chroma" (FLUX sem time_in).
#   CheckpointLoaderSimple → KSampler falha com: 'Chroma' object has no attribute 'time_in'
#   Solução: ChromaDiffusionLoader carrega o modelo de forma compatível com KSampler
RUN git clone --quiet --depth 1 https://github.com/lodestone-rock/ComfyUI_FluxMod.git \
    /comfyui/custom_nodes/ComfyUI_FluxMod

# ─── Patch ComfyUI_FluxMod/loader.py: compatibilidade pick_operations API ─────
# PROBLEMA: ComfyUI_FluxMod chama pick_operations(scaled_fp8=...) mas o
#   ComfyUI atual (main-base, Feb 2026) renomeou o parâmetro para model_config=.
#   Ambas as APIs existiram em versões diferentes:
#     ComfyUI v0.3.x: pick_operations(..., scaled_fp8=None)
#     ComfyUI main (Feb 2026+): pick_operations(..., model_config=None)
# SOLUÇÃO: patch com try/except — tenta a API nova, fallback para antiga.
#   Para Chroma em bf16: scaled_fp8=None mesmo, então ambas as APIs são equivalentes.
RUN python3 << 'EOF'
# NOTA: NÃO importar comfy aqui — este script roda no python3 do sistema (build time),
# não na /opt/venv. O comfy só existe na venv e só é acessível em runtime.
# O inspect.signature check no new_block rodará em runtime dentro do ComfyUI.

path = '/comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/loader.py'
with open(path, 'r') as f:
    content = f.read()

old_pick = '        fp8 = model_conf.optimizations.get("fp8", model_conf.scaled_fp8 is not None)\n        operations = comfy.ops.pick_operations(\n            unet_config.get("dtype"),\n            model.manual_cast_dtype,\n            fp8_optimizations=fp8,\n            scaled_fp8=model_conf.scaled_fp8,\n        )'

new_pick = '        fp8 = model_conf.optimizations.get("fp8", getattr(model_conf, "scaled_fp8", None) is not None)\n        import inspect as _inspect\n        if "scaled_fp8" in _inspect.signature(comfy.ops.pick_operations).parameters:\n            operations = comfy.ops.pick_operations(\n                unet_config.get("dtype"),\n                model.manual_cast_dtype,\n                fp8_optimizations=fp8,\n                scaled_fp8=getattr(model_conf, "scaled_fp8", None),\n            )\n        else:\n            # ComfyUI main (Feb 2026+): uses model_config= instead of scaled_fp8=\n            operations = comfy.ops.pick_operations(\n                unet_config.get("dtype"),\n                model.manual_cast_dtype,\n                fp8_optimizations=fp8,\n                model_config=model_conf,\n            )'

assert old_pick in content, f'ERROR: pick_operations block not found — upstream changed?'
content = content.replace(old_pick, new_pick)
assert '_inspect.signature' in content, 'ERROR: pick_operations patch not applied!'

# FIX 2: strict=False no load_state_dict
# gonzalomoChroma v3.0 foi treinado SEM QKNorm (query_norm/key_norm).
# ComfyUI main adicionou QKNorm ao DoubleStreamBlock — load com strict=True
# falha com "Missing key(s) in state_dict". Com strict=False, RMSNorm inicia
# weight=ones (identidade) — modelo funciona corretamente sem esses pesos.
old_load = '    model.diffusion_model.load_state_dict(state_dict)'
new_load = '    model.diffusion_model.load_state_dict(state_dict, strict=False)  # strict=False: suporte a modelos Chroma sem QKNorm (query_norm/key_norm)'

assert old_load in content, f'ERROR: load_state_dict line not found — upstream changed?'
content = content.replace(old_load, new_load)
assert 'strict=False' in content, 'ERROR: strict=False patch not applied!'

with open(path, 'w') as f:
    f.write(content)

print('loader.py patched OK — pick_operations shim + strict=False applied')
EOF

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

# ─── Patch ComfyUI_FluxMod/layers.py: fix RMSNorm = None ─────────────────────
# PROBLEMA: ComfyUI main setou RMSNorm = None em comfy/ldm/flux/layers.py
#   ("Fix import for some custom nodes, TODO: delete eventually")
#   ComfyUI_FluxMod importa RMSNorm desse módulo e usa em Approximator.__init__:
#     self.norms = nn.ModuleList([RMSNorm(hidden_dim, operations=operations) ...])
#   Com RMSNorm = None, isso vira None(hidden_dim, ...) → TypeError: 'NoneType' object is not callable
# SOLUÇÃO: inserir definição própria de RMSNorm no layers.py ANTES da classe Approximator.
#   IMPORTANTE: o checkpoint usa 'scale' (não 'weight') como nome do parâmetro do RMSNorm.
#   torch.nn.RMSNorm usa 'weight' → state_dict mismatch. Precisamos de 'scale'.
#   A classe abaixo é compatível com o checkpoint do gonzalomoChroma v3.0.
RUN python3 << 'EOF'
path = '/comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/layers.py'
with open(path, 'r') as f:
    content = f.read()

# Insert custom RMSNorm class right before class Approximator
# This overrides the None import from comfy.ldm.flux.layers
# Parameter named 'scale' matches gonzalomoChroma v3.0 checkpoint format
rms_class = '''

# FIX: RMSNorm = None in comfy/ldm/flux/layers.py (ComfyUI main, Feb 2026+)
# Define compatible RMSNorm with 'scale' parameter (matches checkpoint format)
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, dtype=None, device=None, operations=None):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x: Tensor) -> Tensor:
        x_float = x.float()
        rrms = torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x_float * rrms).to(dtype=x.dtype) * self.scale

'''

insert_before = 'class Approximator(nn.Module):'
assert insert_before in content, f'ERROR: Approximator class not found in layers.py'
content = content.replace(insert_before, rms_class + insert_before, 1)
assert 'self.scale = torch.nn.Parameter(torch.ones(dim' in content, 'ERROR: RMSNorm class injection failed!'

with open(path, 'w') as f:
    f.write(content)

print('layers.py patched OK — RMSNorm class defined with scale parameter (checkpoint-compatible)')
EOF

# ─── Patch ComfyUI_FluxMod/nodes.py: fix ChromaPaddingRemoval attention_mask ──
# PROBLEMA: ChromaPromptTruncation.append acessa conditioning[0][1]["attention_mask"]
#   mas o CLIPLoader type="chroma" (ComfyUI T5XXL com return_attention_masks=False)
#   NÃO inclui "attention_mask" no conditioning — KeyError fatal.
# SOLUÇÃO: tornar o uso de attention_mask opcional (graceful fallback).
#   - SE attention_mask presente: trunca tokens até o comprimento real
#   - SE ausente: mantém todos os tokens (sem truncação — leve overhead, seguro)
#   - SEMPRE seta pooled_output=zeros e guidance=0 (obrigatório para Chroma)
RUN python3 << 'EOF'
path = '/comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/nodes.py'
with open(path, 'r') as f:
    content = f.read()

# Find and replace the attention_mask block inside ChromaPromptTruncation.append
old_block = (
    '        pruning_idx = conditioning[0][1]["attention_mask"].sum() + 1\n'
    '        conditioning[0][0] = conditioning[0][0][:, :pruning_idx]\n'
    '        del conditioning[0][1]["attention_mask"]\n'
)
new_block = (
    '        # FIX: attention_mask optional — ComfyUI T5 encoder sem return_attention_masks\n'
    '        if "attention_mask" in conditioning[0][1]:\n'
    '            pruning_idx = conditioning[0][1]["attention_mask"].sum() + 1\n'
    '            conditioning[0][0] = conditioning[0][0][:, :pruning_idx]\n'
    '            del conditioning[0][1]["attention_mask"]\n'
)

assert old_block in content, f'ERROR: attention_mask block not found in nodes.py — upstream changed?'
content = content.replace(old_block, new_block, 1)
assert 'attention_mask" in conditioning[0][1]' in content, 'ERROR: attention_mask patch not applied!'

with open(path, 'w') as f:
    f.write(content)

print('nodes.py patched OK — ChromaPaddingRemoval handles missing attention_mask')
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
    ls /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/nodes.py && echo "FluxMod/ChromaWrapper OK" && \
    grep -q '_inspect.signature' /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/loader.py && echo "loader.py pick_operations patch OK" && \
    grep -q 'strict=False' /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/loader.py && echo "loader.py strict=False patch OK" && \
    grep -q 'lazy-import: FaceAnalysis' /comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py && echo "pulidflux.py patch OK" && \
    grep -q 'self.scale = torch.nn.Parameter' /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/layers.py && echo "layers.py RMSNorm patch OK" && \
    grep -q 'attention_mask" in conditioning' /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/nodes.py && echo "nodes.py ChromaPaddingRemoval patch OK"
