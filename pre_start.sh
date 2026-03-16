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

# Continuar com startup normal
exec /start.sh
