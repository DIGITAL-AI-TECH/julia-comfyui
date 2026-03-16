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
    "timm>=0.9.7,<1.0" \
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

# ─── Patch ComfyUI_FluxMod/layers.py: fix QKNorm — Build #36 ─────────────────
# ROOT CAUSE all-black output confirmado (896x1152 PNG, 3082 bytes, 100% zeros):
# DoubleStreamBlock/SingleStreamBlock herdam do ComfyUI main que ADICIONOU QKNorm
# (query_norm/key_norm via SelfAttention.norm). gonzalomoChroma foi treinado SEM QKNorm.
# Com strict=False (patch anterior), pesos inicializam weight=ones →
#   RMSNorm(x) = x / rms(x) * 1.0  →  Q e K normalizados para norma unitária
#   → dot products Q·K ≈ cos_similarity × head_dim  (muito pequenos para seq longa)
#   → softmax(QK/sqrt(d)) ≈ uniforme (1/N para N~260k tokens)
#   → attention output ≈ mean(V) ≈ próximo de zero
#   → denoising completamente quebrado → imagem all-black.
# FIX: substituir QKNorm por NoOpQKNorm (retorna q, k sem modificação)
#   após super().__init__() em DoubleStreamBlock e SingleStreamBlock.
RUN python3 << 'EOF'
path = '/comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/layers.py'
with open(path, 'r') as f:
    content = f.read()

# 1. Inserir classe NoOpQKNorm antes de DoubleStreamBlock
noop_class = '''
# FIX Build #36: NoOpQKNorm para gonzalomoChroma (treinado SEM QKNorm)
# QKNorm weight=ones normaliza Q/K para norma unitaria → atencao uniforme → all-black
class NoOpQKNorm(torch.nn.Module):
    """Identity: Q e K passam sem normalizacao. Compativel com QKNorm.forward(q,k,v) API."""
    def forward(self, q, k, v):
        return q, k

'''

insert_before = 'class DoubleStreamBlock(layers.DoubleStreamBlock):'
assert insert_before in content, 'ERROR: DoubleStreamBlock class not found in layers.py'
content = content.replace(insert_before, noop_class + insert_before, 1)
assert 'class NoOpQKNorm' in content, 'ERROR: NoOpQKNorm injection failed!'

# 2. DoubleStreamBlock.__init__: disable img_attn.norm e txt_attn.norm
# Chamadas em DoubleStreamBlock.forward: self.img_attn.norm(img_q, img_k, img_v)
#                                         self.txt_attn.norm(txt_q, txt_k, txt_v)
old_double = '        del self.img_mod\n        del self.txt_mod\n'
new_double  = ('        del self.img_mod\n'
               '        del self.txt_mod\n'
               '        # FIX Build #36: desabilitar QKNorm (gonzalomoChroma treinado sem QKNorm)\n'
               '        self.img_attn.norm = NoOpQKNorm()\n'
               '        self.txt_attn.norm = NoOpQKNorm()\n')

assert old_double in content, 'ERROR: del self.img_mod / del self.txt_mod block not found in layers.py'
content = content.replace(old_double, new_double, 1)
assert 'self.img_attn.norm = NoOpQKNorm()' in content, 'ERROR: DoubleStreamBlock QKNorm patch failed!'

# 3. SingleStreamBlock.__init__: disable self.norm
# Chamada em SingleStreamBlock.forward: q, k = self.norm(q, k, v)
old_single = '        del self.modulation\n'
new_single  = ('        del self.modulation\n'
               '        # FIX Build #36: desabilitar QKNorm\n'
               '        self.norm = NoOpQKNorm()\n')

assert old_single in content, 'ERROR: del self.modulation not found in layers.py'
content = content.replace(old_single, new_single, 1)
assert 'self.norm = NoOpQKNorm()' in content, 'ERROR: SingleStreamBlock QKNorm patch failed!'

with open(path, 'w') as f:
    f.write(content)

print('layers.py patched OK — QKNorm disabled in DoubleStreamBlock + SingleStreamBlock (Build #36)')
EOF

# ─── Patch pulidflux.py: forward_orig_fluxmod + forward_orig_chroma_native ───
# Build #37: adicionado forward_orig_chroma_native para UNETLoader (ComfyUI nativo PR #7965)
#
# DOIS CASOS:
#   1. ChromaDiffusionLoader (FluxMod): distribute_modulations(), distill_vec=, skip_mmdit
#      → forward_orig_fluxmod (já existia)
#   2. UNETLoader (comfy.ldm.chroma.model.Chroma): get_modulations(), vec=, skip_mmdit/skip_dit
#      → forward_orig_chroma_native (Build #37 — novo)
#
# Detecção: hasattr(flux_model, 'distribute_modulations') → FluxMod
#           hasattr(flux_model, 'get_modulations') → Chroma nativo
#           else → FLUX.1-dev padrão
RUN python3 << 'PYEOF'
path = '/comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py'
with open(path, 'r') as f:
    content = f.read()

# ─── 1. Inserir forward_orig_fluxmod antes de tensor_to_image ─────────────────
forward_orig_fluxmod = '''
def forward_orig_fluxmod(
    self,
    img,
    img_ids,
    txt,
    txt_ids,
    timesteps,
    guidance=None,
    control=None,
    transformer_options={},
    **kwargs
):
    """FluxMod-compatible forward_orig com PuLID attention injection.
    Usado quando ChromaDiffusionLoader carrega o modelo (arquitetura FluxMod/Chroma).
    Não usa time_in — usa distilled_guidance_layer + distribute_modulations.
    Blocos chamados com distill_vec= (interface FluxMod customizada).
    """
    import torch
    device = comfy.model_management.get_torch_device()
    patches_replace = transformer_options.get("patches_replace", {})

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    img = self.img_in(img)

    # FluxMod: compute modulations via distilled_guidance_layer (no time_in)
    lite = getattr(self, \'lite\', False)
    mod_index_length = 212 if lite else 344
    ts_dim = 8 if lite else 16
    mod_dim = 16 if lite else 32

    distill_timestep = timestep_embedding(timesteps.detach().clone(), ts_dim).to(device=img.device, dtype=img.dtype)
    if guidance is None:
        guidance = torch.zeros_like(timesteps)
    distil_guidance = timestep_embedding(guidance.detach().clone(), ts_dim).to(device=img.device, dtype=img.dtype)

    modulation_index = timestep_embedding(torch.arange(mod_index_length), mod_dim).to(device=img.device, dtype=img.dtype)
    modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
    timestep_guidance = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1)
    input_vec = torch.cat([timestep_guidance, modulation_index], dim=-1)

    mod_vectors = self.distilled_guidance_layer(input_vec)
    mod_vectors_dict = self.distribute_modulations(mod_vectors, len(self.single_blocks), len(self.double_blocks))

    txt = self.txt_in(txt)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    ca_idx = 0
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if i not in getattr(self, \'skip_mmdit\', []):
            img_mod = mod_vectors_dict[f"double_blocks.{i}.img_mod.lin"]
            txt_mod = mod_vectors_dict[f"double_blocks.{i}.txt_mod.lin"]
            double_mod = [img_mod, txt_mod]

            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], pe=args["pe"], distill_vec=args["distill_vec"])
                    return out
                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "pe": pe, "distill_vec": double_mod}, {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, pe=pe, distill_vec=double_mod)

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

            # PuLID attention injection (double blocks)
            if self.pulid_data:
                if i % self.pulid_double_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        condition_start = node_data[\'sigma_start\'] >= timesteps
                        condition_end = timesteps >= node_data[\'sigma_end\']
                        condition = torch.logical_and(condition_start, condition_end).all()
                        if condition:
                            img = img + node_data[\'weight\'] * self.pulid_ca[ca_idx].to(device)(node_data[\'embedding\'], img)
                    ca_idx += 1

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        if i not in getattr(self, \'skip_dit\', []):
            single_mod = mod_vectors_dict[f"single_blocks.{i}.modulation.lin"]

            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], pe=args["pe"], distill_vec=args["distill_vec"])
                    return out
                out = blocks_replace[("single_block", i)]({"img": img, "pe": pe, "distill_vec": single_mod}, {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, pe=pe, distill_vec=single_mod)

            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:, ...] += add

            # PuLID attention injection (single blocks)
            if self.pulid_data:
                real_img = img[:, txt.shape[1]:, ...]
                txt_cur = img[:, :txt.shape[1], ...]
                if i % self.pulid_single_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        condition_start = node_data[\'sigma_start\'] >= timesteps
                        condition_end = timesteps >= node_data[\'sigma_end\']
                        condition = torch.logical_and(condition_start, condition_end).all()
                        if condition:
                            real_img = real_img + node_data[\'weight\'] * self.pulid_ca[ca_idx].to(device)(node_data[\'embedding\'], real_img)
                    ca_idx += 1
                img = torch.cat((txt_cur, real_img), 1)

    img = img[:, txt.shape[1]:, ...]
    final_mod = mod_vectors_dict["final_layer.adaLN_modulation.1"]
    img = self.final_layer(img, distill_vec=final_mod)
    return img

'''

insert_before = 'def tensor_to_image(tensor):'
assert insert_before in content, f'ERROR: tensor_to_image not found in pulidflux.py'
content = content.replace(insert_before, forward_orig_fluxmod + insert_before, 1)
assert 'forward_orig_fluxmod' in content, 'ERROR: forward_orig_fluxmod injection failed!'

# ─── 2. Inserir forward_orig_chroma_native (Build #37 — Chroma via UNETLoader) ─
# Chroma nativo usa: get_modulations(), vec=, skip_mmdit/skip_dit (não distribute_modulations + distill_vec=)
forward_orig_chroma_native = '''
def forward_orig_chroma_native(
    self,
    img,
    img_ids,
    txt,
    txt_ids,
    timesteps,
    guidance=None,
    control=None,
    transformer_options={},
    attn_mask=None,
    **kwargs
):
    """Chroma nativo (UNETLoader ComfyUI) com PuLID attention injection.
    Baseado em comfy/ldm/chroma/model.py forward_orig + injeção PuLID.
    Usa get_modulations() + vec= (diferente do FluxMod que usa distribute_modulations + distill_vec=).
    Build #37.
    """
    import torch
    device = comfy.model_management.get_torch_device()
    transformer_options = transformer_options.copy()
    patches_replace = transformer_options.get("patches_replace", {})

    img = self.img_in(img)

    # distilled vector guidance (igual ao forward_orig nativo do Chroma)
    mod_index_length = 344
    distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(device=img.device, dtype=img.dtype)
    if guidance is None:
        guidance = torch.zeros_like(timesteps)
    distil_guidance = timestep_embedding(guidance.detach().clone(), 16).to(device=img.device, dtype=img.dtype)

    modulation_index = timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(device=img.device, dtype=img.dtype)
    modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
    timestep_guidance_cat = torch.cat([distill_timestep, distil_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1).to(img.dtype).to(img.device)
    input_vec = torch.cat([timestep_guidance_cat, modulation_index], dim=-1)

    mod_vectors = self.distilled_guidance_layer(input_vec)

    txt = self.txt_in(txt)
    ids = torch.cat((txt_ids, img_ids), dim=1)
    pe = self.pe_embedder(ids)

    ca_idx = 0
    blocks_replace = patches_replace.get("dit", {})
    for i, block in enumerate(self.double_blocks):
        if i not in getattr(self, \'skip_mmdit\', []):
            double_mod = (
                self.get_modulations(mod_vectors, "double_img", idx=i),
                self.get_modulations(mod_vectors, "double_txt", idx=i),
            )
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"], out["txt"] = block(img=args["img"], txt=args["txt"],
                                                   vec=args["vec"], pe=args["pe"],
                                                   attn_mask=args.get("attn_mask"))
                    return out
                out = blocks_replace[("double_block", i)](
                    {"img": img, "txt": txt, "vec": double_mod, "pe": pe, "attn_mask": attn_mask},
                    {"original_block": block_wrap})
                txt = out["txt"]
                img = out["img"]
            else:
                img, txt = block(img=img, txt=txt, vec=double_mod, pe=pe, attn_mask=attn_mask)

            if control is not None:
                control_i = control.get("input")
                if i < len(control_i):
                    add = control_i[i]
                    if add is not None:
                        img += add

            # PuLID attention injection (double blocks)
            if self.pulid_data:
                if i % self.pulid_double_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        condition_start = node_data[\'sigma_start\'] >= timesteps
                        condition_end = timesteps >= node_data[\'sigma_end\']
                        condition = torch.logical_and(condition_start, condition_end).all()
                        if condition:
                            img = img + node_data[\'weight\'] * self.pulid_ca[ca_idx].to(device)(node_data[\'embedding\'], img)
                    ca_idx += 1

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        if i not in getattr(self, \'skip_dit\', []):
            single_mod = self.get_modulations(mod_vectors, "single", idx=i)
            if ("single_block", i) in blocks_replace:
                def block_wrap(args):
                    out = {}
                    out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"],
                                       attn_mask=args.get("attn_mask"))
                    return out
                out = blocks_replace[("single_block", i)](
                    {"img": img, "vec": single_mod, "pe": pe, "attn_mask": attn_mask},
                    {"original_block": block_wrap})
                img = out["img"]
            else:
                img = block(img, vec=single_mod, pe=pe, attn_mask=attn_mask)

            if control is not None:
                control_o = control.get("output")
                if i < len(control_o):
                    add = control_o[i]
                    if add is not None:
                        img[:, txt.shape[1]:, ...] += add

            # PuLID attention injection (single blocks)
            if self.pulid_data:
                real_img = img[:, txt.shape[1]:, ...]
                txt_cur  = img[:, :txt.shape[1], ...]
                if i % self.pulid_single_interval == 0:
                    for _, node_data in self.pulid_data.items():
                        condition_start = node_data[\'sigma_start\'] >= timesteps
                        condition_end = timesteps >= node_data[\'sigma_end\']
                        condition = torch.logical_and(condition_start, condition_end).all()
                        if condition:
                            real_img = real_img + node_data[\'weight\'] * self.pulid_ca[ca_idx].to(device)(node_data[\'embedding\'], real_img)
                    ca_idx += 1
                img = torch.cat((txt_cur, real_img), 1)

    img = img[:, txt.shape[1]:, ...]
    if hasattr(self, "final_layer"):
        final_mod = self.get_modulations(mod_vectors, "final")
        img = self.final_layer(img, vec=final_mod)
    return img

'''

content = content.replace(insert_before, forward_orig_chroma_native + insert_before, 1)
assert 'forward_orig_chroma_native' in content, 'ERROR: forward_orig_chroma_native injection failed!'

# ─── 3. Detectar FluxMod vs Chroma nativo vs FLUX.1-dev em apply_pulid_flux ──
# NOTA: old_method deve ser o TEXTO ORIGINAL do pulidflux.py (git clone),
#   NÃO o texto patcheado pelo Build #36. Cada build começa do zero.
old_method = (
    '            # Replace model forward_orig with our own\n'
    '            new_method = forward_orig.__get__(flux_model, flux_model.__class__)\n'
    '            setattr(flux_model, \'forward_orig\', new_method)\n'
)
new_method = (
    '            # Replace model forward_orig with our own\n'
    '            # FIX Build #37: detect Chroma models (no time_in) vs FLUX.1-dev\n'
    '            if hasattr(flux_model, \'distribute_modulations\'):\n'
    '                # FluxMod (ChromaDiffusionLoader) — uses distribute_modulations + distill_vec=\n'
    '                new_method = forward_orig_fluxmod.__get__(flux_model, flux_model.__class__)\n'
    '            elif hasattr(flux_model, \'get_modulations\') and not hasattr(flux_model, \'time_in\'):\n'
    '                # Chroma nativo (UNETLoader, ComfyUI PR #7965) — uses get_modulations + vec=\n'
    '                new_method = forward_orig_chroma_native.__get__(flux_model, flux_model.__class__)\n'
    '            else:\n'
    '                # FLUX.1-dev padrão\n'
    '                new_method = forward_orig.__get__(flux_model, flux_model.__class__)\n'
    '            setattr(flux_model, \'forward_orig\', new_method)\n'
)

assert old_method in content, f'ERROR: forward_orig FluxMod detection block not found — upstream changed?'
content = content.replace(old_method, new_method, 1)
assert 'forward_orig_chroma_native' in content and 'get_modulations' in content, 'ERROR: Chroma native detection patch not applied!'

with open(path, 'w') as f:
    f.write(content)

print('pulidflux.py patched OK — Build #37: forward_orig_chroma_native + 3-way detection (FluxMod/ChromaNative/FLUX)')
PYEOF

# ─── ReActorFaceSwap ──────────────────────────────────────────────────────────
# Faceswap tradicional: troca rosto de referência no output gerado
# Deps obrigatórias: onnxruntime-gpu ✅, insightface ✅ (já instalados)
#
# setuptools<81: CRÍTICO — ReActor usa pkg_resources (removido em setuptools>=81)
#   sem este pin, o auto-installer do ReActor quebra no startup do ComfyUI
#
# opencv-python-headless: variante sem GUI — correta para container headless
# opencv-python (com GUI) é mutuamente exclusivo com headless — compartilham cv2
#   o uninstall pós-clone garante que o auto-installer do ReActor não reinstale
#   a versão errada (requirements.txt do ReActor pede opencv-python, sem headless)
#
# NÃO usar -r requirements.txt do ReActor: reinstalaria insightface+onnxruntime
#   e poderia causar downgrade/conflito com as versões já fixadas acima
RUN /opt/venv/bin/pip install --quiet --no-cache-dir "setuptools<81" && \
    /opt/venv/bin/pip install --quiet --no-cache-dir "opencv-python-headless>=4.7.0.72"

# Deps necessárias para ReActor registrar seus nodes no ComfyUI
# DIAGNÓSTICO Build #41: nodes.py tem `from segment_anything import sam_model_registry`
#   no nível de módulo → ModuleNotFoundError impede o node de registrar (silent skip)
# segment-anything: importado no nível de módulo em nodes.py — CRÍTICO
# albumentations, ultralytics, onnx: também usados em scripts/ e nodes.py
RUN /opt/venv/bin/pip install --quiet --no-cache-dir \
    "albumentations>=1.4.16" \
    "segment-anything" \
    "ultralytics" \
    "onnx>=1.14.0"

# REPO RENOMEADO: comfyui-reactor-node → ComfyUI-ReActor (URL antiga = privada/arquivada)
# --depth=1: clone raso (mais rápido, menos download)
RUN git clone --depth=1 https://github.com/Gourieff/ComfyUI-ReActor.git \
    /comfyui/custom_nodes/comfyui-reactor-node

# ultralytics pode ter substituído opencv-python-headless por opencv-python (GUI)
# Garantir que apenas headless existe: uninstall GUI → reinstalar headless
RUN /opt/venv/bin/pip uninstall -y opencv-python 2>/dev/null || true && \
    /opt/venv/bin/pip install --quiet --no-cache-dir "opencv-python-headless>=4.7.0.72"

# Silenciar warnings do albumentations update checker (sem impacto funcional)
ENV NO_ALBUMENTATIONS_UPDATE=1

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
    ln -sf /runpod-volume/models/checkpoints        /comfyui/models/diffusion_models  && \
    ln -sf /runpod-volume/models/facerestore_models /comfyui/models/facerestore_models && \
    ln -sf /runpod-volume/models/upscale_models     /comfyui/models/upscale_models     && \
    ln -sf /runpod-volume/models/animatediff        /comfyui/models/animatediff        && \
    ln -sf /runpod-volume/models/controlnet         /comfyui/models/controlnet         && \
    ln -sf /runpod-volume/models/loras              /comfyui/models/loras              && \
    ln -sf /runpod-volume/models/vhs_video          /comfyui/models/vhs_video
# NOTA: diffusion_models → checkpoints para ChromaDiffusionLoader encontrar gonzalomoChroma_v30.safetensors
# animatediff → motion modules para AnimateDiff Evolved
# controlnet  → ControlNet models (OpenPose, DepthAnything, etc.)
# loras       → LoRA models
# vhs_video   → VideoHelperSuite cache

# ─── HuggingFace cache → volume (evita re-download de modelos ControlNet/DWPose) ─
ENV HF_HOME=/runpod-volume/huggingface

# ─── Video & Animation Stack ──────────────────────────────────────────────────
# Deps Python para os novos custom nodes de vídeo e pose
# NOTA: controlnet_aux lista opencv-python → instalar via headless no cleanup final
# mediapipe: pose detection fallback (não faz CUDA JIT no module import)
# diffusers>=0.33.0: necessário para Wan Video 2.1 wrapper
# gguf>=0.17.1: suporte a modelos quantizados Wan/LLM
# NO numba: não está em nenhum requirements dos nodes adicionados
RUN /opt/venv/bin/pip install --quiet --no-cache-dir \
    "diffusers>=0.33.0" \
    "accelerate>=1.2.1" \
    "peft>=0.17.0" \
    "sentencepiece>=0.2.0" \
    "protobuf" \
    "pyloudnorm" \
    "gguf>=0.17.1" \
    "imageio-ffmpeg" \
    "scikit-image>=0.21" \
    "scikit-learn" \
    "mediapipe" \
    "omegaconf" \
    "trimesh" \
    "addict" \
    "yacs" \
    "matplotlib" \
    "python-dateutil" \
    "yapf"

# ─── AnimateDiff Evolved ──────────────────────────────────────────────────────
# Animação SD 1.5 com motion modules — base para dance videos
# Suporta: motion modules, MotionLoRA, SparseCtrl, IPAdapter + AnimateDiff
RUN git clone --quiet --depth 1 https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git \
    /comfyui/custom_nodes/ComfyUI-AnimateDiff-Evolved

# ─── ControlNet Aux — preprocessors de pose e profundidade ────────────────────
# CRÍTICO para replicar Kling-style dance:
#   DWPose: extração de pose corporal 2D frame-a-frame de vídeos de referência
#   OpenPose: alternativa a DWPose (rosto + mãos + corpo)
#   DepthAnything: mapa de profundidade para composição 3D
#   Lineart/Canny: controle de estilo/bordas
# Pipeline dance: video_ref → DWPose → pose sequence → AnimateDiff/WanVideo + ControlNet
RUN git clone --quiet --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git \
    /comfyui/custom_nodes/comfyui_controlnet_aux

# ─── Video Helper Suite ────────────────────────────────────────────────────────
# I/O de vídeo: load_video → frames, frames → export_video
# Necessário para: carregar vídeo de referência de dança, exportar resultado
RUN git clone --quiet --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    /comfyui/custom_nodes/ComfyUI-VideoHelperSuite

# ─── Wan Video Wrapper (kijai) — melhor vídeo open-source 2025 ────────────────
# Wan 2.1: melhor modelo open-source para geração de vídeo com motion realista
# Qualidade ~80% do Kling para dance/motion com modelos 14B (28GB VRAM)
# Cabe confortavelmente no A100 80GB junto com ControlNet e IPAdapter
# Modelos no volume: /runpod-volume/models/wan_video/
RUN git clone --quiet --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git \
    /comfyui/custom_nodes/ComfyUI-WanVideoWrapper

# ─── Ultimate SD Upscale — tile upscale sem distorção ─────────────────────────
# Upscale em tiles: evita distorção em imagens grandes (>2048px)
# Indispensável para imagens finais 4K com faces nítidas
RUN git clone --quiet --depth 1 https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git \
    /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale

# Garantir opencv headless após todos os installs (vários nodes instalam GUI opencv)
RUN /opt/venv/bin/pip uninstall -y opencv-python 2>/dev/null || true && \
    /opt/venv/bin/pip install --quiet --no-cache-dir "opencv-python-headless>=4.7.0.72"

# Symlink Wan Video models
RUN ln -sf /runpod-volume/models/wan_video /comfyui/models/wan_video 2>/dev/null || true

# ─── VideoOutputBridge — expõe VHS_VideoCombine gifs como images ───────────────
# PROBLEMA: runpod/worker-comfyui handler captura apenas campo 'images' (SaveImage)
#   VHS_VideoCombine salva em campo 'gifs' que o handler não forwarda
#   → job retorna 'success_no_images' mesmo com vídeo gerado
# SOLUÇÃO: VideoOutputBridge remapeia 'gifs' → 'images' no histórico do ComfyUI
#   permite que o handler capture .mp4/.webm/.gif como se fossem imagens
# REF: https://github.com/arthurtravers/ComfyUI-VideoOutputBridge
RUN git clone --quiet --depth 1 https://github.com/arthurtravers/ComfyUI-VideoOutputBridge.git \
    /comfyui/custom_nodes/ComfyUI-VideoOutputBridge

# ─── ComfyUI-Impact-Pack — FaceDetailer + SAMLoader ───────────────────────────
# Necessário para gonzaLomo Flux Refiner v3.0 workflow
# Fornece: FaceDetailer, SAMLoader, BboxDetectorSEGS, etc.
# REF: https://github.com/ltdrdata/ComfyUI-Impact-Pack
RUN git clone --quiet --depth 1 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git \
    /comfyui/custom_nodes/ComfyUI-Impact-Pack && \
    cd /comfyui/custom_nodes/ComfyUI-Impact-Pack && \
    /opt/venv/bin/pip install --quiet --no-cache-dir -r requirements.txt 2>/dev/null || true && \
    /opt/venv/bin/pip install --quiet --no-cache-dir segment-anything

# ─── rgthree-comfy — Power Lora Loader ────────────────────────────────────────
# Necessário para gonzaLomo Flux Refiner v3.0 workflow
# Fornece: Power Lora Loader, Context nodes
# REF: https://github.com/rgthree/rgthree-comfy
RUN git clone --quiet --depth 1 https://github.com/rgthree/rgthree-comfy.git \
    /comfyui/custom_nodes/rgthree-comfy && \
    cd /comfyui/custom_nodes/rgthree-comfy && \
    /opt/venv/bin/pip install --quiet --no-cache-dir -r requirements.txt 2>/dev/null || true

# ─── Verificação de build ──────────────────────────────────────────────────────
RUN /opt/venv/bin/python -c "import onnxruntime, insightface, timm, facexlib, einops, kornia, cv2; print('onnxruntime ' + onnxruntime.__version__); print('providers: ' + str(onnxruntime.get_available_providers())); print('insightface OK'); print('timm OK'); print('facexlib OK'); print('opencv ' + cv2.__version__)" && \
    ls /comfyui/custom_nodes/ComfyUI_IPAdapter_plus/IPAdapterPlus.py && echo "IPAdapter OK" && \
    ls /comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py && echo "PuLID OK" && \
    ls /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/nodes.py && echo "FluxMod/ChromaWrapper OK" && \
    ls /comfyui/custom_nodes/comfyui-reactor-node && echo "ReActor OK" && \
    ls /comfyui/custom_nodes/ComfyUI-AnimateDiff-Evolved && echo "AnimateDiff OK" && \
    ls /comfyui/custom_nodes/comfyui_controlnet_aux && echo "ControlNet Aux OK" && \
    ls /comfyui/custom_nodes/ComfyUI-VideoHelperSuite && echo "VideoHelperSuite OK" && \
    ls /comfyui/custom_nodes/ComfyUI-WanVideoWrapper && echo "WanVideo OK" && \
    ls /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale && echo "UltimateUpscale OK" && \
    grep -q '_inspect.signature' /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/loader.py && echo "loader.py pick_operations patch OK" && \
    grep -q 'strict=False' /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/loader.py && echo "loader.py strict=False patch OK" && \
    grep -q 'lazy-import: FaceAnalysis' /comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py && echo "pulidflux.py patch OK" && \
    grep -q 'self.scale = torch.nn.Parameter' /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/layers.py && echo "layers.py RMSNorm patch OK" && \
    grep -q 'class NoOpQKNorm' /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/layers.py && echo "layers.py QKNorm fix (Build #36) OK" && \
    grep -q 'attention_mask" in conditioning' /comfyui/custom_nodes/ComfyUI_FluxMod/flux_mod/nodes.py && echo "nodes.py ChromaPaddingRemoval patch OK" && \
    grep -q 'forward_orig_fluxmod' /comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py && echo "pulidflux.py forward_orig_fluxmod patch OK" && \
    grep -q 'distribute_modulations' /comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py && echo "pulidflux.py FluxMod detection patch OK" && \
    ls /comfyui/custom_nodes/ComfyUI-Impact-Pack/modules/impact/core.py && echo "Impact-Pack OK" && \
    ls /comfyui/custom_nodes/rgthree-comfy/nodes/power_lora_loader.py && echo "rgthree Power Lora Loader OK"

# ─── ComfyUI handler check_server timeout ─────────────────────────────────────
# O handler.py usa process_status (PID file) para decidir entre:
#   - PID file encontrado + processo vivo → poll infinito
#   - PID file NÃO encontrado → COMFY_API_FALLBACK_MAX_RETRIES (default 500 × 50ms = 25s)
# Template env vars (COMFY_API_AVAILABLE_MAX_RETRIES=3600) parecem não ser aplicadas no
# serverless — ENVs do Dockerfile são aplicadas mas o INTERVAL_MS permanece em 50ms.
# FIX DEFINITIVO v2 — handler.py timeout patch via Python (mais robusto que sed)
#
# Contexto: handler.py usa check_server(url, COMFY_API_AVAILABLE_MAX_RETRIES, COMFY_API_AVAILABLE_INTERVAL_MS)
#   - COMFY_API_AVAILABLE_MAX_RETRIES default = 0 → sem PID file → fallback 500 × 50ms = 25s (FAIL)
#   - ENV vars do Dockerfile/template não chegam corretamente no serverless RunPod
#   - Abordagem DUAL: Python patch modifica valores + append overrides como backup
#
# Resultado: COMFY_API_FALLBACK_MAX_RETRIES=7200 × INTERVAL=1000ms = 7200s = 2h (mais que suficiente)
#   (ComfyUI leva ~3-4min para iniciar com todos os custom nodes)
RUN python3 - << 'PYEOF'
import re

with open('/handler.py', 'r') as f:
    content = f.read()

# 1) Increase FALLBACK_MAX_RETRIES: 500 → 7200
old = 'COMFY_API_FALLBACK_MAX_RETRIES = 500'
new = 'COMFY_API_FALLBACK_MAX_RETRIES = 7200'
if old in content:
    content = content.replace(old, new)
    print(f'✅ FALLBACK_MAX_RETRIES: 500 → 7200')
else:
    print(f'⚠️  FALLBACK_MAX_RETRIES pattern not found')

# 2) Change interval default: 50 → 1000 ms (single or multiline variants)
for old_ms in [
    "os.environ.get(\"COMFY_API_AVAILABLE_INTERVAL_MS\", 50)",
    "os.environ.get('COMFY_API_AVAILABLE_INTERVAL_MS', 50)",
]:
    new_ms = old_ms.replace(", 50)", ", 1000)")
    if old_ms in content:
        content = content.replace(old_ms, new_ms)
        print(f'✅ INTERVAL_MS default: 50 → 1000')
        break
else:
    print(f'⚠️  INTERVAL_MS pattern not found')

with open('/handler.py', 'w') as f:
    f.write(content)

print('Verification:')
for line in content.splitlines():
    if 'COMFY_API_FALLBACK' in line or 'COMFY_API_AVAILABLE_INTERVAL_MS' in line:
        print(f'  {line.strip()}')
PYEOF

# Backup: append module-level overrides AFTER all existing code
# Python evaluates module globals top-to-bottom; last assignment wins at call time
RUN printf '\n\n# ── RunPod patch: override retry constants ──────────────────────────────────\nCOMFY_API_FALLBACK_MAX_RETRIES = 7200\nCOMFY_API_AVAILABLE_INTERVAL_MS = 1000\nCOMFY_API_AVAILABLE_MAX_RETRIES = 7200\n' >> /handler.py && \
    echo "handler.py final constants:" && \
    grep -n "COMFY_API_FALLBACK\|COMFY_API_AVAILABLE_MAX_RETRIES\|COMFY_API_AVAILABLE_INTERVAL" /handler.py | tail -20

ENV COMFY_API_AVAILABLE_MAX_RETRIES=7200
ENV COMFY_API_AVAILABLE_INTERVAL_MS=1000

# ─── Pre-start download script ────────────────────────────────────────────────
# Baixa modelos grandes (T5 encoder 10.8 GB) diretamente para o volume
# no primeiro cold start — bypassa S3 API quota completamente
COPY pre_start.sh /pre_start.sh
RUN chmod +x /pre_start.sh

CMD ["/pre_start.sh"]
