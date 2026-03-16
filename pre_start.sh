#!/usr/bin/env bash
# pre_start.sh — Baixa modelos grandes direto para o volume antes de iniciar ComfyUI
# Roda apenas se o arquivo não existir (one-time download por volume)

# AnimateDiff Evolved (ADE) espera animatediff_models/ mas Dockerfile cria animatediff/
# Criar symlink adicional para garantir compatibilidade
ln -sf /runpod-volume/models/animatediff /comfyui/models/animatediff_models 2>/dev/null || true

# ── WanVideoWrapper (kijai) — symlinks de modelos ─────────────────────────────
# WanVideoModelLoader procura em diffusion_models/ (symlink de checkpoints/)
# → adicionar Wan2.1-T2V-1.3B como entrada em checkpoints para aparecer em diffusion_models
ln -sf /runpod-volume/models/wan_video/Wan2.1-T2V-1.3B \
    /runpod-volume/models/checkpoints/Wan2.1-T2V-1.3B 2>/dev/null || true

# WanVideoVAELoader procura em vae/
ln -sf /runpod-volume/models/wan_video/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth \
    /runpod-volume/models/vae/Wan2.1_VAE.pth 2>/dev/null || true

# LoadWanVideoT5TextEncoder procura em text_encoders/ (pasta não existe por padrão)
mkdir -p /comfyui/models/text_encoders 2>/dev/null || true
ln -sf /runpod-volume/models/wan_video/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \
    /comfyui/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth 2>/dev/null || true

T5="/runpod-volume/models/wan_video/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"

if [ ! -f "$T5" ]; then
    echo "worker-comfyui: T5 encoder não encontrado. Baixando 10.8 GB diretamente para o volume..."
    mkdir -p "$(dirname $T5)"
    wget -q --show-progress --progress=dot:giga \
        --header "Authorization: Bearer ${HF_TOKEN}" \
        -O "$T5" \
        "https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"
    echo "worker-comfyui: T5 download completo: $(ls -lh $T5)"
else
    echo "worker-comfyui: T5 encoder já existe ($(ls -lh $T5 | awk '{print $5}')). Pulando download."
fi

# ── GonzaLomo SDXL Unity XL DMD (v4.0) ───────────────────────────────────────
# SDXL checkpoint para refinamento + geração direta (6.6 GB)
# CivitAI model 1513492 / version 1943922
GONZALOMO_SDXL="/runpod-volume/models/checkpoints/gonzalomoXLFluxPony_v40UnityXLDMD.safetensors"
if [ ! -f "$GONZALOMO_SDXL" ]; then
    echo "worker-comfyui: GonzaLomo SDXL Unity DMD não encontrado. Baixando 6.6 GB..."
    wget -q --show-progress --progress=dot:giga \
        --header "Authorization: Bearer ${CIVITAI_TOKEN}" \
        -O "$GONZALOMO_SDXL" \
        "https://civitai.com/api/download/models/1943922?token=${CIVITAI_TOKEN}"
    echo "worker-comfyui: GonzaLomo SDXL download completo: $(ls -lh $GONZALOMO_SDXL)"
else
    echo "worker-comfyui: GonzaLomo SDXL já existe ($(ls -lh $GONZALOMO_SDXL | awk '{print $5}')). Pulando download."
fi

# ── GonzaLomo FLUX SAIO (v1.0) — base para Flux Refiner workflow ──────────────
# FLUX checkpoint base para gonzaLomo Flux Refiner v3.0 (16 GB)
# CivitAI model 1513492 / version 1968729
GONZALOMO_FLUX="/runpod-volume/models/checkpoints/gonzalomoXLFluxPony_v10FluxSAIO.safetensors"
if [ ! -f "$GONZALOMO_FLUX" ]; then
    echo "worker-comfyui: GonzaLomo FLUX SAIO não encontrado. Baixando 16 GB..."
    wget -q --show-progress --progress=dot:giga \
        --header "Authorization: Bearer ${CIVITAI_TOKEN}" \
        -O "$GONZALOMO_FLUX" \
        "https://civitai.com/api/download/models/1968729?token=${CIVITAI_TOKEN}"
    echo "worker-comfyui: GonzaLomo FLUX SAIO download completo: $(ls -lh $GONZALOMO_FLUX)"
else
    echo "worker-comfyui: GonzaLomo FLUX SAIO já existe ($(ls -lh $GONZALOMO_FLUX | awk '{print $5}')). Pulando download."
fi

# ── SAM ViT-B (Impact-Pack / FaceDetailer) ────────────────────────────────────
# Modelo SAM necessário para SAMLoader node do ComfyUI-Impact-Pack
# Usado no gonzaLomo Flux Refiner v3.0 para detecção de face no FaceDetailer
SAM_MODEL="/runpod-volume/models/sams/sam_vit_b_01ec64.pth"
mkdir -p "$(dirname $SAM_MODEL)"
ln -sf "$(dirname $SAM_MODEL)" /comfyui/models/sams 2>/dev/null || true
if [ ! -f "$SAM_MODEL" ]; then
    echo "worker-comfyui: SAM ViT-B não encontrado. Baixando ~375 MB..."
    wget -q --show-progress --progress=dot:mega \
        -O "$SAM_MODEL" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    echo "worker-comfyui: SAM ViT-B download completo: $(ls -lh $SAM_MODEL)"
else
    echo "worker-comfyui: SAM ViT-B já existe ($(ls -lh $SAM_MODEL | awk '{print $5}')). Pulando download."
fi

# ── YOLO face detector (Impact-Pack / FaceDetailer / UltralyticsDetectorProvider) ─
# face_yolov8m.pt — necessário para UltralyticsDetectorProvider no gonzaLomo Flux Refiner v3.0
# Impact-Pack procura em comfyui/models/ultralytics/bbox/
mkdir -p /runpod-volume/models/ultralytics/bbox
ln -sf /runpod-volume/models/ultralytics /comfyui/models/ultralytics 2>/dev/null || true
YOLO_FACE="/runpod-volume/models/ultralytics/bbox/face_yolov8m.pt"
if [ ! -f "$YOLO_FACE" ]; then
    echo "worker-comfyui: face_yolov8m.pt não encontrado. Baixando ~24 MB..."
    wget -q --show-progress --progress=dot:mega \
        -O "$YOLO_FACE" \
        "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt"
    echo "worker-comfyui: face_yolov8m.pt download completo: $(ls -lh $YOLO_FACE)"
else
    echo "worker-comfyui: face_yolov8m.pt já existe ($(ls -lh $YOLO_FACE | awk '{print $5}')). Pulando download."
fi

# Continuar com startup normal
exec /start.sh
